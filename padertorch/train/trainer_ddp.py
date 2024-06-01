"""
    This module contains the Trainer class which can be used to train
    configurable padertorch models.
"""
import os
import sys
import contextlib
import itertools
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import functools
import collections
import dataclasses
from datetime import timedelta

import numpy as np
import torch
import torch.nn
import warnings
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast as amp_autocast
from torch.cuda.amp import GradScaler

from paderbox.utils.nested import deflatten
import padertorch as pt
from padertorch.configurable import Configurable
from padertorch.train.optimizer import Optimizer, Adam
from padertorch.train.runtime_tests import test_run
from padertorch.train.hooks import *

__all__ = [
    'Trainer',
    'InteractiveTrainer',
]


@dataclasses.dataclass
class DDP_Config:
    # dist_backend: str = "nccl"
    # dist_url: str = "env://"
    world_size: int = None
    rank: int = None
    device_id: int = None
    timeout: int = os.environ.get('TIMEOUT_DDP', 1800)

    def __post_init__(self):
        self.timeout = int(self.timeout)


class DummyWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_audio(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def add_figure(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


def get_world_size():
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = torch.cuda.device_count()
        # raise ValueError("Missing world size")
    return world_size


def get_rank():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
    else:
        raise ValueError("Missing rank index")

    return rank


def get_ddp_config(rank):
    world_size = get_world_size()
    device_id = rank % torch.cuda.device_count()
    ddp_config = DDP_Config(
        world_size=world_size, rank=rank, device_id=device_id
    )
    return ddp_config


def setup_distributed(rank, dist_backend, dist_url):
    world_size = get_world_size()
    config = get_ddp_config(rank)
    if "RANK" in os.environ:
        dist.init_process_group(
            backend=dist_backend,
            init_method=dist_url,
            timeout=timedelta(seconds=config.timeout),
        )
    elif "SLURM_PROCID" in os.environ:
        dist.init_process_group(
            backend=dist_backend,
            init_method=dist_url,
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=config.timeout),
        )
    else:
        raise ValueError("Missing rank index")


class Trainer(Configurable):

    @classmethod
    def finalize_dogmatic_config(cls, config):
        if 'optimizer' not in config.keys():
            config['optimizer'] = {'factory': Adam}

    def __init__(
            self,
            model: 'pt.Model',
            storage_dir,
            optimizer,
            loss_weights=None,
            summary_trigger=(1, 'epoch'),
            checkpoint_trigger=(1, 'epoch'),
            stop_trigger=(1, 'epoch'),
            virtual_minibatch_size=1,
            dist_backend='nccl',
            dist_url='env://',
            autocast=False,
            compile=False,
    ):
        """

        Args:
            model: a `padertorch.base.Model` object
            storage_dir: The structure of produced storage_dir is:
                .
                ├── checkpoints
                │   ├── ckpt_7122.pth
                │   ├── ckpt_14244.pth
                │   ├── ckpt_best_loss.pth -> ckpt_7122.pth
                │   ├── ckpt_latest.pth -> ckpt_14244.pth
                │   └── ckpt_ranking.json
                ├── events.out.tfevents.1548851867.ntsim5
            optimizer: a `padertorch.train.optimizer.Optimizer` object
                or dict of Optimizers
            loss_weights: dict of weights for model with multiple losses
            summary_trigger: `pytorch.train.trigger.IntervalTrigger` object
                or tuple describing the interval when summaries
                are written to event files.
                See padertorch.train.hooks.SummaryHook for a description of
                what a summary is.
            checkpoint_trigger: `padertorch.train.trigger.IntervalTrigger`
                object or tuple describing the interval when checkpoints
                are saved. See padertorch.train.hooks.CheckpointHook and
                padertorch.train.hooks.ValidationHook for a description of
                what happens on a checkpoint.
            stop_trigger: `padertorch.train.trigger.EndTrigger` object
                or tuple describing the endpoint of the training
            virtual_minibatch_size: Runs the optimisation in
                virtual_minibatch_size steps. By default run it after each
                review call.
                The advantage of a virtual_minibatch_size over addressing a
                minibatch dimension in forward and review is a lower memory
                footprint on cost of cpu time.
                Note: The gradients are accumulated and not averaged.
                Note: The virtual_minibatch_size is fixed and can contain data
                    from two epochs.


        Usage:

            # For test_run we recommend to do it without prefetch
            trainer = Trainer(...)  # or: Trainer.from_config(...)
            trainer.test_run(tr_ds, val_ds)
            trainer.train(tr_ds.prefetch(4, 8), val_ds_with.prefetch(4, 8))

        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                'Expect that the model is a subclass from padertorch.Module.\n'
                f'Got: type: {type(model)}\n{model}'
            )

        if compile:
            model = torch.compile(model)

        self.model = model
        self.ddp_model = None

        assert isinstance(optimizer, Optimizer), optimizer
        optimizer.set_parameters(model.parameters())

        self.optimizer = optimizer

        self.storage_dir = Path(storage_dir).expanduser().resolve()
        self.writer = None
        self.train_timer = ContextTimerDict()
        self.validate_timer = ContextTimerDict()
        self.iteration = -1
        self.epoch = -1

        self.loss_weights = loss_weights
        self.virtual_minibatch_size = virtual_minibatch_size

        self.dist_backend = dist_backend
        self.dist_url = dist_url
        self.world_size = get_world_size()
        self.ddp_config = get_ddp_config(get_rank())
        self.is_master = True

        self.autocast = autocast
        if self.autocast:
            self.grad_scaler = GradScaler()

        self.hooks = [
            SummaryHook(summary_trigger),
            CheckpointHook(checkpoint_trigger),
            StopTrainingHook(stop_trigger),
        ]
        self._summary_trigger = summary_trigger
        self._stop_trigger = stop_trigger
        self._checkpoint_trigger = checkpoint_trigger

        import tensorboardX  # The import is slow -> lazy import
        self.writer_cls = tensorboardX.SummaryWriter

    def dist_init(self, rank):
        setup_distributed(rank, self.dist_backend, self.dist_url)
        self.ddp_config = get_ddp_config(rank)

        self.is_master = (self.ddp_config.rank == 0)

        model = self.model
        model.to(self.ddp_config.device_id)
        model = DistributedDataParallel(model)
        self.model = model.module
        self.ddp_model = model

        if not self.is_master:
            self.hooks.pop(0)  # Remove SummaryHook
            self.hooks.pop(0)  # Remove CheckpointHook
            self.writer_cls = DummyWriter
            pop_ind = []
            # Remove ValidationHook
            for i, hook in enumerate(self.hooks):
                if isinstance(hook, ValidationHook):
                    pop_ind.append(i)
            for i in pop_ind:
                self.hooks.pop(i)
                for j, _ in enumerate(pop_ind):
                    pop_ind[j] -= 1

    def test_run(
            self,
            train_iterator,
            validation_iterator,
            device=0 if torch.cuda.is_available() else 'cpu',
            *,
            test_with_known_iterator_length=False,
            temporary_directory=None,
            deterministic_atol=1e-5,
            deterministic_rtol=1e-5,
            loss_atol=1e-6,
            loss_rtol=1e-6,
            virtual_minibatch_size=None,
    ):
        """
        Run a test on the trainer instance (i.e. model test).

        Also tests weather validation step is deterministic.
        # ToDo: is the following still true? are there any other restrictions?
        !!Does not work with layers changing their internal state such as BatchNorm!!

        Tests:
         - forward (train and validate)
         - deterministic output in eval
         - simple review dict test


        Args:
            train_iterator:
            validation_iterator:
            device:
            test_with_known_iterator_length:
            temporary_directory:
                Specify a path as alternative to tempfile.TemporaryDirectory().
                Note: This directory will not be deleted and it is expected that
                it is empty.
                Usecase: Fast debugging of the reports to tensorboard.
                         After the test run you can start tensorboard and inspect
                         the reported values.

        """
        test_run(
            self,
            train_iterator,
            validation_iterator,
            device=device,
            test_with_known_iterator_length=test_with_known_iterator_length,
            temporary_directory=temporary_directory,
            deterministic_atol=deterministic_atol,
            deterministic_rtol=deterministic_rtol,
            loss_atol=loss_atol,
            loss_rtol=loss_rtol,
            virtual_minibatch_size=virtual_minibatch_size,
        )

    def train_ddp(
        self,
        rank,
        train_dataset,
        validation_dataset,
        resume=False,
        progress_bar=True,
        track_emissions=False,
        device=None,
        exist_ok=False,
    ):
        # Load checkpoint before initializing the distributed process group
        if resume:
            assert resume is True, resume
            self.load_checkpoint()
            resume = False
            exist_ok = True,
        else:
            if self.is_master:
                assert not self.checkpoint_dir.exists(),\
                    f'A checkpoint directory already exists. If you want to ' \
                    f'restart the training set resume to True.'
            self.iteration = 0
            self.epoch = 0

        for hook in self.hooks:
            if isinstance(hook, ValidationHook):
                hook.iterator = validation_dataset

        self.dist_init(rank)
        self.train(
            train_dataset,
            resume=resume,
            progress_bar=progress_bar,
            track_emissions=track_emissions,
            device=device,
            exist_ok=exist_ok,
        )

    def train(
            self,
            train_dataset,
            *,
            progress_bar=True,
            track_emissions=False,
            resume=False,
            device=None,
            exist_ok=False,
    ):
        """
        A simplified training loop::

            for epoch in range(1, ...):
                for example in train_iterator:
                    model_out = self.model(example)
                    review = self.model.review(example, model_out)
                    review = maybe_add_loss_from_losses(review)
                    review['loss'].backward()
                    self.optimizer.step()
                    add_review_to_tensorboardX(review)

        The remaining code takes care about calling validation and save the
        result to tensorboard (if a validation_hook is registered), save
        checkpoints, cleanup checkpoints that are stale (not best according
        to metric and not last) and display a progessbar.
        The code is designed that many aspects can be customized.
        (e.g. see test_runtime_tests.py DictTrainer for multi model trainer)

        Args:
            train_dataset:
                The train_dataset is python iterable (e.g. tuple, list, ...)
                that can consumed multiple times (i.e. not generator).

                Usually it will be paderbox.database.BaseIterator that is
                returned from a database in paderbox.database.
            progress_bar: flag whether to show a progress bar or not.
            track_emissions: flag whether to track emissions using codecarbon.
            resume:
                Whether to resume a training or start a fresh one.
            device:
                TODO: remove, needed for test_run
        """

        assert torch.cuda.is_available()

        if resume:
            assert resume is True, resume
            self.load_checkpoint()
        else:
            if not exist_ok:
                if self.is_master:
                    assert not self.checkpoint_dir.exists(),\
                        'A checkpoint directory already exists. '\
                        'If you want to restart the training set resume to '\
                        'True.'
                self.iteration = 0
                self.epoch = 0
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

        # Change model to train mode (e.g. activate dropout)
        self.ddp_model.train()

        self._to(self.ddp_config.device_id)

        # Reset all gradients
        self.optimizer_zero_grad()

        self.writer = self.writer_cls(str(self.storage_dir))
        hooks = [*self.hooks]
        if progress_bar and self.is_master:
            try:
                max_it_len = len(train_dataset)
            except TypeError:
                # TypeError: object of type '...' has no len()
                max_it_len = None
            progressbar_hook = ProgressBarHook(self._stop_trigger, max_it_len)
            # set_last updates the iteration counter in case of resume
            progressbar_hook.set_last(self.iteration, self.epoch)
            hooks.append(progressbar_hook)
        if track_emissions:
            raise NotImplementedError()
            # hooks.append(EmissionsTrackerHook(
            #     self._summary_trigger, storage_dir=self.storage_dir))
        hooks = sorted(hooks, key=lambda h: h.priority, reverse=True)


        # ================ MAIN TRAINING LOOP! ===================
        try:
            train_iterable = None
            while True:
                new_epoch = False
                if train_iterable is None:
                    new_epoch = True

                    # Call pre_step between the epochs.
                    # We call it here, so it is done, before the iteration
                    # over the train_dataset starts.
                    for hook in hooks:
                        hook.pre_step(self)

                    train_iterable = iter(train_dataset)

                optimize = True
                with self.train_timer['time_per_iteration'] as timer:
                    for minibatch_index in range(self.virtual_minibatch_size):
                        with self.train_timer['time_per_data_loading']:
                            example = list(itertools.islice(train_iterable, 1))
                            if len(example) == 0:
                                train_iterable = None
                                self.epoch += 1
                                if minibatch_index == 0:
                                    optimize = False
                                break  # end minibatch loop

                        if new_epoch:
                            new_epoch = False
                        elif minibatch_index == 0:
                            # Call pre_step after getting the next example,
                            # to correctly detect the next epoch
                            with timer.pause():
                                for hook in hooks:
                                    hook.pre_step(self)

                        assert len(example) == 1, (len(example), example)
                        example = example[0]

                        loss, example, model_output, review = \
                            self.train_step(self.ddp_model, example, self.ddp_config.device_id)

                        with timer.pause():
                            for hook in hooks:
                                hook.post_step(self, example, model_output, review)

                        # Release pytorch object to reduce memory footprint
                        del example
                        del model_output
                        del review

                        with self.train_timer['time_per_backward']:
                            loss.backward(retain_graph=False)
                        del loss


                    # Only the summary hook will use optimizer_review
                    if optimize:
                        with self.train_timer['time_per_optimize']:
                            optimizer_summary = self.optimizer_step()
                            for hook in hooks:
                                hook.post_optimize(self, optimizer_summary)
                            del optimizer_summary

                        self.iteration += 1

        except StopTraining:
            pass
        finally:
            try:
                # dist.destroy_process_group()
                for hook in hooks:
                    hook.close(self)
            except Exception:
                print('Exception in finally. May hide actual exception!!!\n'
                      'You may comment this finally block for debugging.')
                raise
            self.writer.close()
            self.writer = None

    _non_validation_start_time = None

    def validate(self, validation_iterator):
        """
        used by ValidationHook

        :param validation_iterator:
        :return:
        """
        validation_start_time = self.validate_timer.timestamp()

        if self._non_validation_start_time is not None:
            self.validate_timer.timings['non_validation_time'].append(
                validation_start_time - self._non_validation_start_time
            )

        # Disable backward mode with `no_grad()`.
        with self.validate_timer['validation_time'], torch.no_grad():
            # Change model to eval mode (e.g. deactivate dropout).
            self.ddp_model.eval()
            try:
                validation_iter = iter(validation_iterator)
                while True:
                    with self.validate_timer['time_per_iteration']:
                        try:
                            with self.validate_timer['time_per_data_loading']:
                                example = next(validation_iter)
                        except StopIteration:
                            break
                        step_output = self.validation_step(
                            self.ddp_model, example, self.ddp_config.device_id)
                    yield step_output
                    del example, step_output

            finally:
                self.ddp_model.train()
                self._non_validation_start_time = self.validate_timer.timestamp()

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        if self.autocast:
            self.grad_scaler.unscale_(self.optimizer.optimizer)

        summary = self.clip_grad({})

        # Add learning rate to the summary
        for i, param_group in enumerate(self.optimizer.optimizer.param_groups):
            summary['scalars'][f'lr/param_group_{i}'] = param_group['lr']

        # Do the actual optimization
        if self.autocast:
            self.grad_scaler.step(self.optimizer.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

        self.optimizer_zero_grad()
        return summary

    def train_step(self, model, example, device):
        return self.step(model, example, self.train_timer, device)

    def validation_step(self, model, example, device):
        # [1:] -> ignore the loss. Is already in scalars.
        return self.step(model, example, self.validate_timer, device)[1:]

    def step(self, model, example, timer, device):
        with amp_autocast(enabled=self.autocast):
            try:
                # TODO: Backup OutOfMemory
                with timer['time_per_to_device']:
                    if isinstance(model, DistributedDataParallel):
                        example = model.module.example_to_device(
                            example, device
                        )
                    else:
                        example = model.example_to_device(
                            example, device
                        )
                with timer['time_per_forward']:
                    model_out = model(example)
                with timer['time_per_review']:
                    if isinstance(model, DistributedDataParallel):
                        review = model.module.review(example, model_out)
                    else:
                        review = model.review(example, model_out)
                    loss, summary = self._review_to_loss_and_summary(review)
                    if self.autocast:
                        loss = self.grad_scaler.scale(loss)
                    return loss, example, model_out, summary
            except Exception:
                data = {
                    'model': self.model,
                    'state_dict': self.state_dict(),
                    'example': example,
                }
                if 'model_out' in locals():
                    data['model_out'] = model_out
                if 'review' in locals():
                    data['review'] = review

                log_path_pattern = self.log_error_state(data)
                print(f'Wrote\n{log_path_pattern}\nfor debugging.')
                raise

    def _review_to_loss_and_summary(self, review):
        """

        Splits the review to the loss and the summary.
        The review contains a "loss" key or a "losses" key.
        The loss key contains the loss itself, while the losses is a dictionary
        and it is combined with the loss_weights,
        i.e. sum(loss_weights[k] * losses[k] for k in losses)

        The losses are added to the scalars to be logged.
        The loss is always logged as loss in the scalars.

        """

        if 'scalars' not in review:
            review['scalars'] = {}

        if 'losses' in review:
            assert 'loss' not in review, review
            losses = review['losses']

            loss = 0.
            loss_weights = self.loss_weights
            if len(losses) != 1:
                if loss_weights is None:
                    raise Exception(
                        'You can not have multiple losses without specifying '
                        f'loss_weights. losses: {losses}'
                    )
                elif set(loss_weights.keys()) != set(losses.keys()):
                    import textwrap
                    from IPython.lib.pretty import pretty
                    raise Exception(
                        'You can not have multiple losses without specifying '
                        'a loss_weight for each loss.'
                        f'\nlosses:'
                        f'\n{textwrap.indent(pretty(losses), " "*4)}'
                        f'\nloss_weights:\n'
                        f'{textwrap.indent(pretty(loss_weights), " "*4)}'
                    )

            for key, value in losses.items():
                weight = loss_weights[key] if loss_weights is not None else 1.
                if weight != 0:
                    loss = loss + (weight * value)
                review['scalars'][key] = value.item()
                review['scalars'][f'{key}_loss_weight'] = weight
            del review['losses']
            # review['loss'] = loss
        else:
            assert 'loss' in review, review
            loss = review.pop('loss')

        review['scalars']['loss'] = loss.item()

        assert loss.dim() == 0, loss

        if not torch.isfinite(loss):
            # Write each interesting object to an individual file, because
            # not each object is serializable with `torch.save`.
            log_path_pattern = self.log_error_state({
                'model': self.model,
                'state_dict': self.state_dict(),
                'review': review,
            })
            raise RuntimeError(
                f"The loss ({loss}) is not finite.\n"
                f"See error states (model, example, model_out and review) in "
                f"{log_path_pattern}."
            )

        return loss, review

    def log_error_state(self, data_dict, folder='log', file=sys.stdout):
        """

        Args:
            data_dict:

        Returns:
            log_path_pattern that describes the successfully written files.

        """
        import paderbox as pb

        import pickle

        class MyPickleModule:
            __name__ = 'MyPickleModule'  # Pytorch tests if name is dill

            class Pickler(pickle._Pickler):

                def save(self, obj, save_persistent_id=True):
                    try:
                        super().save(obj, save_persistent_id=save_persistent_id)
                    except Exception as e:
                        print(f'Cannot pickle {obj!r}, replace it with a str.', file=file)
                        super().save(repr(obj), save_persistent_id=save_persistent_id)

            # Not sure, when this happens, but when `torch.save` uses
            # `_legacy_save`, the MyPickleModule needs a dump function.
            # Reported from TCL.
            def dump(self, obj, file, protocol=None, *, fix_imports=True):
                # copy from pickle source code
                self.Pickler(file, protocol, fix_imports=fix_imports).dump(obj)

        written = []
        for k, v in data_dict.items():
            p = self.storage_dir / folder / f'error_state_{k}.pth'
            p.parent.mkdir(exist_ok=True)
            try:
                # Not every object can be serialized.
                with pb.io.atomic.open_atomic(p, 'wb') as fd:
                    torch.save(v, fd, pickle_module=MyPickleModule())
                written.append(k)
            except Exception as e:
                import traceback
                log_file = (self.storage_dir / folder / f'{k}.log')
                with log_file.open('w') as fd:
                    traceback.print_exc(file=fd)
                print(f'Cannot save {k}. {type(e)}: {e}. See {log_file}', file=file)

        written = ','.join(written)
        return str(self.storage_dir / folder / f'error_state_{{{written}}}.pth')

    def register_hook(self, hook):
        if isinstance(hook, (tuple, list)):
            for h in hook:
                self.register_hook(h)
        else:
            self.hooks.append(hook)

    def register_validation_hook(
            self, validation_iterator, metric='loss', maximize=False,
            max_checkpoints=1, n_back_off=0, lr_update_factor=1 / 10,
            back_off_patience=None, early_stopping_patience=None
    ):
        """

        Args:
            validation_iterator:
            metric:
                The metric that is used for deciding which checkpoints are
                kept. The key must be 'loss', a key in review['losses']
                or a key in review['scalars'].
            maximize: if True the metric has to be maximized else minimized.
            max_checkpoints: The number of checkpoints to keep.
                When max_checkpoints is None, keep all checkpoints.
            n_back_off: number of times the best checkpoint is reloaded to
                continue training with an updated learning rate.
            lr_update_factor: the factor by which the lr is multiplied in case
                of back off. Should be smaller than 1.
            back_off_patience: the number of allowed degradations before
                backing off
            early_stopping_patience: the number of allowed degradations before
                stopping training. Should be larger than back_off_patience.


        Returns:

        """
        if self.is_master:
            self.register_hook(BackOffValidationHook(
                trigger=self._checkpoint_trigger,
                iterator=validation_iterator,
                metric=metric,
                maximize=maximize,
                max_checkpoints=max_checkpoints,
                n_back_off=n_back_off,
                lr_update_factor=lr_update_factor,
                back_off_patience=back_off_patience,
                early_stopping_patience=early_stopping_patience,
            ))

    def clip_grad(self, summary: dict):
        # TODO: report clipped and unclipped
        # TODO: allow clip=None but still report grad_norm

        summary.setdefault('scalars', {})
        summary.setdefault('histograms', {})

        def check(grad_norm):
            if not np.all(np.isfinite(pt.utils.to_numpy(grad_norm, detach=True))):
                # Write each interesting object to an individual file, because
                # not each object is serializable with `torch.save`.
                log_path_pattern = self.log_error_state({
                    'model': self.model,
                    'state_dict': self.state_dict(),
                    'optimizer_summary': summary,
                    'grad': {k: v.grad for k, v in self.model.named_parameters()},
                })
                raise RuntimeError(
                    f"The grad_norm ({grad_norm}) is not finite.\n"
                    f"See error states (model, example, model_out and review) in "
                    f"{log_path_pattern}."
                )

        grad_norm = self.optimizer.clip_grad()
        check(grad_norm)
        summary['scalars'][f'grad_norm'] = grad_norm
        summary['histograms'][f'grad_norm_'] = \
            torch.Tensor([grad_norm])

        return summary

    @property
    def checkpoint_dir(self):
        return self.storage_dir / 'checkpoints'

    def default_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / f'ckpt_{self.iteration}.pth'

    def state_dict(self):
        optimizer_state_dict = self.optimizer.state_dict()
        state_dict = dict(
                model=self.model.state_dict(),
                iteration=self.iteration,
                epoch=self.epoch,
                optimizer=optimizer_state_dict,
                hooks=dict(),
        )
        for hook in self.hooks:
            if hook is not self.model:
                hook_state = hook.state_dict()
                if hook_state is not None:
                    assert hook.uid not in state_dict['hooks'], (hook.uid, state_dict['hooks'].keys())
                    state_dict['hooks'][hook.uid] = hook_state
        return state_dict

    def save_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.default_checkpoint_path()

        torch.save(
            self.state_dict(),
            str(checkpoint_path)
        )

        # Create relative symlink to latest checkpoint
        latest_symlink_path = (checkpoint_path.parent / f'ckpt_latest.pth').absolute()
        if latest_symlink_path.is_symlink():
            latest_symlink_path.unlink()
        latest_symlink_path.symlink_to(checkpoint_path.name)

        print(f"{datetime.now()}: Saved model and optimizer state "
              f"at iteration {self.iteration} to {checkpoint_path}")

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

        self.iteration = state_dict['iteration']
        self.epoch = state_dict['epoch']

        if 'hooks' in state_dict:
            hook_states = state_dict['hooks']
            for hook in self.hooks:
                hook.set_last(self.iteration, self.epoch)
                if hook.uid in hook_states:
                    hook.load_state_dict(hook_states.pop(hook.uid))
            assert len(hook_states) == 0, hook_states.keys()
        else:
            warnings.warn(
                "You are resuming an old checkpoint which does not include "
                "hook states. If you want to recover hook states you have to "
                "add them manually to the checkpoint prior to resumption."
            )

    def load_checkpoint(self, map_location='cpu'):
        checkpoint_path = self.checkpoint_dir / 'ckpt_latest.pth'
        assert checkpoint_path.is_file(), checkpoint_path

        checkpoint_dict = torch.load(
            str(checkpoint_path), map_location=map_location
        )

        if torch.__version__ == "1.12.0":
            warnings.warn("This torch version (1.12.0) has a bug, for more information"
                          " see https://github.com/pytorch/pytorch/issues/80809,"
                          " the capturable flag of the parameter groups in the optimizer"
                          " state_dicts will be set to True.")
            optimizer_state_dict = checkpoint_dict['optimizer']
            param_groups = optimizer_state_dict['param_groups']
            for group in param_groups:
                group['capturable'] = True

        self.load_state_dict(checkpoint_dict)

        print(f"Loaded checkpoint '{checkpoint_path}' (iteration {self.iteration})")

    def _to(self, device):
        if device is None:
            # Do nothing
            return

        if device == 'cpu' or isinstance(device, int):
            # single device: e.g. 'cpu', 0, 1, ...
            pass
        else:
            raise ValueError(device)

        device = torch.device(device)

        print(f'Move trainer, model and optimizer to {device}.')

        self.ddp_model.to(device)
        self.model.to(device)
        self.optimizer.to(device)

    def cpu(self):
        return self._to('cpu')

    def cuda(self, device=None):
        assert device is None or isinstance(device, int), device
        if device is None:
            device = torch.device('cuda')
        return self._to(device)


class MultiDeviceTrainer(Trainer):
    """

    A Trainer that does not change the model device.
    The losses may be located on different devices, so this trainer moves all
    losses to the cpu.

    Note: The device argument of the Trainer.train is used to move the example
          to the device.
    """

    def _review_to_loss_and_summary(self, review):
        if 'losses' in review:
            review['losses'] = {
                k: v.cpu()
                for k, v in review['losses'].items()
            }
        return super()._review_to_loss_and_summary(review)

    def to(self, device):
        pass


class ContextTimerDict:
    """
    To be able to keep the measurements, we need to create the object before.
    Then each measurement can be started with a context manager.

    >>> np.set_printoptions(precision=2)
    >>> timer = ContextTimerDict()
    >>> with timer['test']:
    ...     time.sleep(0.1)
    >>> with timer['test']:
    ...     time.sleep(0.1)
    >>> with timer['test_2']:
    ...     time.sleep(0.1)
    >>> for _ in timer('test_3', range(3)):
    ...     time.sleep(0.1)

    Ignore timing when an exception is raised
    >>> with contextlib.suppress(Exception), timer['test_2']:
    ...     raise Exception

    >>> timer  # doctest: +SKIP
    ContextTimerDict: {'test': array([0.1, 0.1]), 'test_2': array([0.1]), 'test_3': array([1.96e-06, 4.80e-06, 3.87e-06])}
    >>> d = timer.as_dict
    >>> for k, v in d.items():
    ...     v = [f'{e:.2f}' for e in v]
    ...     print(f'{k}: {v}')
    test: ['0.10', '0.10']
    test_2: ['0.10']
    test_3: ['0.00', '0.00', '0.00']


    >>> timer = ContextTimerDict()
    >>> with timer['test'] as t:
    ...     time.sleep(0.1)
    ...     with t.pause():
    ...         time.sleep(0.1)
    ...     time.sleep(0.1)
    >>> timer
    ContextTimerDict: {'test': array([0.2])}
"""
    def __init__(self):
        self.timestamp = time.perf_counter  # time.process_time
        self.timings = defaultdict(list)
        self.clear()

    def clear(self):
        self.timings.clear()

    class Excluder:
        def __init__(self, timestamp):
            self.duration = []
            self.timestamp = timestamp

        @contextlib.contextmanager
        def pause(self):
            start = self.timestamp()
            yield
            end = self.timestamp()
            self.duration.append(end -start)


    @contextlib.contextmanager
    def __getitem__(self, item):
        assert isinstance(item, str), item
        start = self.timestamp()
        excluder = self.Excluder(self.timestamp)
        yield excluder
        end = self.timestamp()
        self.timings[item].append(end - start - sum(excluder.duration))

    @property
    def as_dict(self):
        return {k: np.array(time) for k, time in self.timings.items()}

    def __repr__(self):
        return f'{self.__class__.__name__}: ' + repr(self.as_dict)

    def __str__(self):
        return str(self.as_dict)

    def __call__(self, key, iterable):
        iterator = iter(iterable)

        class StopIterationIgnoredByContextlib(Exception):
            pass
            # contextlib.contextmanager tries to inform the user with a
            # DeprecationWarning because of PEP 479.
            # The cas here is still conform with PEP 479 (i.e. use __future__
            # import).
            # To suppress the warning, convert StopIteration to this Exception
            # and catch it.

        try:
            while True:
                with self[key]:
                    try:
                        example = next(iterator)
                    except StopIteration:
                        raise StopIterationIgnoredByContextlib
                yield example
        except StopIterationIgnoredByContextlib:
            pass


class InteractiveTrainer(Trainer):
    def __init__(
            self,
            model,
            optimizer,
            loss_weights=None,
            stop_trigger=(200, 'epoch'),
            summary_trigger=(50, 'epoch'),
            validation_trigger=None,
    ):
        super().__init__(
            model=model,
            storage_dir='this/is/no/path',
            optimizer=optimizer,
            loss_weights=loss_weights,
            summary_trigger=summary_trigger,
            checkpoint_trigger=summary_trigger,
            stop_trigger=stop_trigger,
        )
        del self.hooks[1]
        assert len(self.hooks) == 2, self.hooks
        # Trainer uses checkpoint_trigger as validation_trigger
        if validation_trigger is None:
            self.validation_trigger = summary_trigger
        else:
            self.validation_trigger = validation_trigger
        self.writer_cls = lambda x: InteractiveWriter()


    @functools.wraps(Trainer.train)
    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        return self.writer


class InteractiveWriter:
    def __init__(self):
        self.scalars = collections.defaultdict(list)

    def add_scalar(self, tag, scalar_value, global_step, walltime=None):
        if tag.split('/')[0] in ['training_timings', 'validation_timings']:
            return
        print(f'{global_step}, {tag}: {scalar_value}')

        walltime = time.time() if walltime is None else walltime
        self.scalars[tag].append({
            'value': scalar_value,
            'global_step': global_step,
            'walltime': walltime,
        })

    def add_audio(self, tag, snd_tensor, global_step,
                  sample_rate=44100, walltime=None):
        pass

    def add_image(self, tag, img_tensor, global_step, walltime=None):
        pass

    def add_histogram(self, tag, values, global_step,
                      bins='tensorflow', walltime=None):
        pass

    def close(self):
        pass

# TODO: write function for those to functions outside of trainer
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
