""" This module contains various hooks which perform actions during training.

Hooks replace a huge amount of conditions in the trainer code.
Having individual hook modules allows to enable and disable specific
functionality.

E.g., adding a learning rate schedule without adding further conditions to the
trainer.

"""
import json
from collections import defaultdict
from enum import IntEnum
from pathlib import Path
import types

import numpy as np
import progressbar
import torch
import tensorboardX

import paderbox as pb
from padertorch.train.trigger import IntervalTrigger, EndTrigger

__all__ = [
    'SummaryHook',
    'SimpleCheckpointHook',
    'ValidationHook',
    'CheckpointedValidationHook',
    'ProgressBarHook',
    'StopTrainingHook',
    'StopTraining',
    'CKPT_EXT',
]


CKPT_EXT = 'pth'


class Priority(IntEnum):
    """
    Summary 50
    Print 40 NotImplemented
    ProgressBar(TQDM) 30 NotImplemented
    Validation 25
    Checkpoint 20
    End 10

    End has to be the last one
    Summary before Validation, clears timer information
    Print and ProgressBar may access Summary
    """
    END = 10
    DEFAULT = 15
    VALIDATION = 20
    CHECKPOINT = 25
    PROGRESS = 30
    PRINT = 40
    SUMMARY = 50


class Hook:
    @property
    def priority(self):
        return Priority.DEFAULT

    def pre_step(self, trainer: 'pt.Trainer'):
        """
        function is called before each iteration of the train iterator

        Args:
            trainer:

        Returns:

        """
        pass

    def post_step(self, trainer: 'pt.Trainer', example, model_output,
                  review):
        """
        function is called after each train step

        Args:
            trainer:
            example:
            model_output:
            review:

        Returns:
        """
        pass

    def close(self, trainer: 'pt.Trainer'):
        pass

    def set_last(self, iteration, epoch):
        pass


class TriggeredHook(Hook):

    def __init__(self, trigger=None):
        """

        Args:
            trigger: tuple or Trigger.
                When Tuple, the first entry is the trigger interval length and
                the second the unit (i.e. 'epoch' or 'iteration').
                Example: (1, 'epoch')
        """
        self.trigger = IntervalTrigger.new(trigger)

    def set_last(self, iteration, epoch):
        self.trigger.set_last(iteration, epoch)


class SummaryHook(TriggeredHook):
    """
    Responsible to write a summary in the tfevents file.
    The tfevents can be visualised in the tensorboard.

    The summary consists of the returned scalars, images, audios, etc of the
    training that are returned by the model review function.
    Note: It does not contain the learned model parameters, they are saved at
    the checkpoints.

    To save results of the validation refer to ValidationHook.
    """

    def __init__(
            self,
            trigger,
            writer: tensorboardX.SummaryWriter,
            summary_prefix='training',
    ):
        super().__init__(trigger)
        self.reset_summary()
        self.summary_prefix = summary_prefix
        self.writer = writer

    @property
    def priority(self):
        return Priority.SUMMARY

    @staticmethod
    def empty_summary_dict():
        # MappingProxyType is similar to a frozen dict (does not exist)
        #   Ensures that no key is added.
        return types.MappingProxyType(dict(
            # losses=defaultdict(list),
            scalars=defaultdict(list),
            histograms=defaultdict(list),
            audios=dict(),
            images=dict(),
            texts=dict(),
            figures=dict(),
            timings=dict(),
        ))

    def reset_summary(self):
        # Todo: add figures
        self.summary = self.empty_summary_dict()

    def update_summary(self, review):
        allowed_keys = {
            'loss',
            'losses',
            'scalars',
            'histograms',
            'audios',
            'images',
            'texts',
            'figures',
        }
        redundant_keys = set(review.keys()) - allowed_keys
        assert len(redundant_keys) == 0, (redundant_keys, review.keys(), allowed_keys)

        poped_review = {**review}  # copy for "pop"

        # note item is the pytorch function to get the value of a tensor
        self.summary['scalars']['loss'].append(poped_review.pop('loss').item())
        for key, loss in poped_review.pop('losses', dict()).items():
            self.summary['scalars'][key].append(loss.item())
        for key, scalars in poped_review.pop('scalars', dict()).items():
            self.summary['scalars'][key].extend(self._to_list(scalars))
        for key, histogram in poped_review.pop('histograms', dict()).items():
            self.summary['histograms'][key].extend(self._to_list(histogram))
            # do not hold more than 1M values in memory
            self.summary['histograms'][key] = \
                self.summary['histograms'][key][-1000000:]
        for key, audio in poped_review.pop('audios', dict()).items():
            self.summary['audios'][key] = audio  # snapshot
        for key, image in poped_review.pop('images', dict()).items():
            self.summary['images'][key] = image  # snapshot
        for key, figure in poped_review.pop('figures', dict()).items():
            self.summary['figures'][key] = figure  # snapshot
        for key, text in poped_review.pop('texts', dict()).items():
            assert isinstance(text, str), text
            self.summary['texts'][key] = text  # snapshot

        assert len(poped_review) == 0, (poped_review, review)

    @staticmethod
    def _to_list(scalars):
        if torch.is_tensor(scalars):
            scalars = scalars.clone().cpu().data.numpy()
        if isinstance(scalars, np.ndarray):
            scalars = scalars.flatten().tolist()
        if not isinstance(scalars, (list, tuple)):
            assert np.isscalar(scalars)
            scalars = [scalars]
        return scalars

    def compute_timings(self, trainer):
        timer_dict = trainer.timer.as_dict
        # Special handling for time_per_data_loading and time_per_train_step
        #  Calculate
        #   - time_per_step: time of loading plus train step per example
        #   - time_rel_data_loading: time_for_loading / time_per_step
        #   - time_rel_train_step: time_train_step / time_per_step
        # Note: It is not guarantied that the size of time_per_data_loading and
        #       time_per_train_step is equal, because the Summary Hook is
        #       called between dataloading and train step. So the loading can
        #       be part of the previous summary, while the train step is in the
        #       next summary.
        time_per_data_loading = timer_dict.pop('time_per_data_loading', [0])
        time_per_train_step = timer_dict.pop('time_per_train_step', [0])
        time_per_train_step_to_device = timer_dict.pop('time_per_train_step_to_device', [0])
        time_per_train_step_forward = timer_dict.pop('time_per_train_step_forward', [0])
        time_per_train_step_review = timer_dict.pop('time_per_train_step_review', [0])
        time_per_backward = timer_dict.pop('time_per_backward', [0])

        summary_timings = {}
        time_per_step = (
                np.mean(time_per_data_loading) + np.mean(time_per_train_step)
        )
        if time_per_step > 0:
            summary_timings['time_per_step'] = time_per_step

            sum_time_per_train_step = np.sum(time_per_train_step)
            sum_time_per_data_loading = np.sum(time_per_data_loading)
            sum_time_per_train_step_to_device = np.sum(time_per_train_step_to_device)
            sum_time_per_train_step_forward = np.sum(time_per_train_step_forward)
            sum_time_per_train_step_review = np.sum(time_per_train_step_review)
            sum_time_per_backward = np.sum(time_per_backward)

            total_train_time = (
                    sum_time_per_data_loading + sum_time_per_train_step
            )
            summary_timings['time_rel_data_loading'] = \
                sum_time_per_data_loading / total_train_time
            summary_timings['time_rel_train_step'] = \
                sum_time_per_train_step / total_train_time
            if sum_time_per_train_step > 0:
                summary_timings['time_rel_to_device'] = \
                    sum_time_per_train_step_to_device / sum_time_per_train_step
                summary_timings['time_rel_forward'] = \
                    sum_time_per_train_step_forward / sum_time_per_train_step
                summary_timings['time_rel_review'] = \
                    sum_time_per_train_step_review / sum_time_per_train_step
                summary_timings['time_rel_backward'] = \
                    sum_time_per_backward / sum_time_per_train_step
        summary_timings.update({
            key: timing.mean() for key, timing in timer_dict.items()
        })
        trainer.reset_timer()
        return summary_timings

    def finalize_summary(self, trainer):
        assert len(self.summary['timings']) == 0, self.summary['timings']
        for key, timing in self.compute_timings(trainer).items():
            self.summary['timings'][key] = timing
        self.summary = trainer.model.modify_summary(self.summary)

    def dump_summary(self, trainer: 'pt.Trainer'):
        iteration = trainer.iteration
        prefix = self.summary_prefix

        time_prefix = f'{prefix}_timings'

        for key, scalar in self.summary['scalars'].items():
            self.writer.add_scalar(
                f'{prefix}/{key}', scalar, iteration)
        for key, scalar in self.summary['timings'].items():
            self.writer.add_scalar(
                f'{time_prefix}/{key}', scalar.mean(), iteration)
        for key, histogram in self.summary['histograms'].items():
            self.writer.add_histogram(
                f'{prefix}/{key}', np.array(histogram), iteration
            )
        for key, audio in self.summary['audios'].items():
            if isinstance(audio, (tuple, list)):
                assert len(audio) == 2, (len(audio), audio)
                self.writer.add_audio(
                    f'{prefix}/{key}', audio[0],
                    iteration, sample_rate=audio[1]
                )
            else:
                self.writer.add_audio(
                    f'{prefix}/{key}', audio,
                    iteration, sample_rate=16000
                )
        for key, image in self.summary['images'].items():
            self.writer.add_image(f'{prefix}/{key}', image, iteration)
        for key, text in self.summary['texts'].items():
            self.writer.add_text(f'{prefix}/{key}', text, iteration)
        for key, figure in self.summary['figures'].items():
            self.writer.add_figure(f'{prefix}/{key}', figure, iteration)

        self.reset_summary()

    def pre_step(self, trainer: 'pt.Trainer'):
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch) \
                and trainer.iteration != 0:
            self.finalize_summary(trainer)
            self.dump_summary(trainer)

    def post_step(self, trainer: 'pt.Trainer', example, model_out, review):
        self.update_summary(review)

    def close(self, trainer: 'pt.Trainer'):
        self.finalize_summary(trainer)
        self.dump_summary(trainer)
        self.writer.close()


class SimpleCheckpointHook(TriggeredHook):
    """ Can be used to keep all checkpoints, e.g. for continuous evaluation
            (keep_all = False) or to only store the most recent checkpoint
            (keep_all = True).
            Cannot be used together with a CheckpointedValidationHook
    """
    def __init__(self, trigger, keep_all=False):
        super().__init__(trigger)
        self.keep_all = keep_all
        self.last_checkpoint_path = None

    @property
    def priority(self):
        return Priority.CHECKPOINT

    def pre_step(self, trainer: 'pt.Trainer'):
        checkpoint_path = trainer.default_checkpoint_path()
        trainer.save_checkpoint(checkpoint_path)
        if not(self.keep_all) and self.last_checkpoint_path.exists():
            self.last_checkpoint_path.unlink()
        self.last_checkpoint_path = checkpoint_path


class ValidationHook(SummaryHook):
    """
    Responsible to do the validation and write its results into the
    tfevents file. The tfevents can be visualised in the tensorboard.

    The saved data consists of the returned scalars, images, audios, etc
    of the validation that are returned from the model review function.
    Note: It does not contain the learned model parameters, they are save at
    the checkpoints.
    """
    def __init__(self, trigger, iterator, writer, lr_scheduler=None):
        super().__init__(trigger, summary_prefix='validation',
                         writer=writer)
        self.iterator = iterator
        self.lr_scheduler = lr_scheduler

    @property
    def priority(self):
        return Priority.VALIDATION

    def pre_step(self, trainer: 'pt.Trainer'):
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch):
            assert all([len(value) == 0 for value in self.summary.values()])
            assert len(trainer.timer.timings) == 0, trainer.timer
            print('Starting Validation')
            at_least_one_value = False
            for model_out, review in trainer.validate(self.iterator):
                at_least_one_value = True
                if self.lr_scheduler is not None:
                    if isinstance(self.lr_scheduler, dict):
                        review['scalars'].update({
                            f'{module}_learning_rate_{i}': param_group['lr']
                            for module, scheduler in self.lr_scheduler.items()
                            for i, param_group in enumerate(
                                scheduler.optimizer.param_groups
                            )
                        })
                    else:
                        review['scalars'].update({
                            f'learning_rate_{i}': param_group['lr']
                            for i, param_group in enumerate(
                                self.lr_scheduler.optimizer.param_groups
                            )
                        })
                self.update_summary(review)
            if not at_least_one_value:
                raise Exception(
                    f'Got an empty validation iterator: {self.iterator}'
                )
            self.finalize_summary(trainer)
            mean_loss = np.mean(self.summary['scalars']['loss'])
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, dict):
                    for scheduler in self.lr_scheduler.values():
                        scheduler.step(self.summary['scalars'], trainer.epoch)
                else:
                    self.lr_scheduler.step(
                        self.summary['scalars'], trainer.epoch
                    )
            self.dump_summary(trainer)
            assert len(trainer.timer.timings) == 0, trainer.timer
            print(f'Finished Validation. Mean loss: {mean_loss}')

    def post_step(self, trainer: 'pt.Trainer', example, model_out, review):
        pass

    def close(self, trainer: 'pt.Trainer'):
        self.writer.close()


class _Metric:
    """ Bookkeeping of metrics (comparison, best value, checkpoint path,
        symlink) needed for CheckpointedValidationHook.
    """
    def __init__(self, metric_key, criterion, checkpoint_dir):
        self._key = metric_key
        self._criterion = criterion
        self._checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self._symlink_name = f'ckpt_best_{metric_key}.{CKPT_EXT}'

        assert criterion in ('min', 'max'), criterion
        # self._value = float('inf') if criterion == 'min' else -float('inf')
        self._value = None

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self._key!r}, {self._criterion!r}, '
            f'best={self.resolved_symlink_path}'
            f')'
        )

    @property
    def name(self):
        return self._key

    @property
    def paths(self):
        return ([self.resolved_symlink_path]
                if self.resolved_symlink_path.exists() else [])

    @property
    def values(self):
        if self._value is None:
            return []
        else:
            return [self._value]

    def is_better(self, value):
        """ Decides whether current metric value is better than best
            previous one. Has to work for cost and gain objectives
            => See init for details.
        """
        if self._value is None:
            return True
        elif self._criterion == 'min':
            return value < self._value
        elif self._criterion == 'max':
            return value > self._value
        else:
            raise AssertionError(f'Should not ne reachable: {self._criterion}')

    def update(self, value, checkpoint_path):
        """ Update to best metric value, corresponding checkpoint path
            and set symlink to checkpoint.
        """
        self._value = value
        # create relative symlink to best checkpoint for metric
        symlink_path = self.symlink_path
        if symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(checkpoint_path.name)

    @property
    def symlink_path(self):
        """
        The absolute path to the symlink "file".
        """
        return (self._checkpoint_dir / self._symlink_name).absolute()

    @property
    def resolved_symlink_path(self):
        """
        The absolute path to the file on that the symlink points.
        Note: returns the symlink itself, when the symlink does not exists.
        """
        return (self._checkpoint_dir / self._symlink_name).resolve()

    def to_json(self):
        """ Dump metric state information into dictionary. """
        return dict(key=self._key,
                    criterion=self._criterion,
                    values=self.values,
                    paths=[path.relative_to(self._checkpoint_dir)
                           for path in self.paths])

    def set_state(self, state_dict):
        """ Set state of the metrics object from a state dictionary.
            This is usually useful when the state_dict stems from a json file
            and was previously created by to_json.
        """
        assert self._key == state_dict['key'], (self._key, state_dict['key'])
        assert self._criterion == state_dict['criterion'], (
            self._criterion, state_dict['criterion'])
        assert len(state_dict['paths']) == len(state_dict['values']), \
            state_dict
        if len(state_dict['paths']) == 0:
            pass
        elif len(state_dict['paths']) == 1:
            expect = self._checkpoint_dir / state_dict['paths'][0]
            assert self.resolved_symlink_path == expect, (
                self.resolved_symlink_path, expect)
            self._value = state_dict['values'][0]
        else:
            assert False, state_dict['paths']


class CheckpointedValidationHook(ValidationHook):
    """ Performs model validation and keeps checkpoints for model states that
        perform best on a given set of metrics.
        Cannot be used together with a ValidationHook
        or a SimpleCheckpointHook.

    Checkpointing and validation:
     - save checkpoint
     - validate and collect summary
     - update best checkpoints according to metrics
     - dump summary to tfevents file
     - cleanup stale checkpoints
     - save CheckpointedValidationHook state in `_json_filename`

    """
    _json_filename = 'ckpt_state.json'

    def __init__(
            self,
            trigger,
            iterator,
            writer,
            checkpoint_dir: Path,
            metrics=None,
            lr_scheduler=None,
            keep_all=False,
            init_from_json=False,
    ):
        super().__init__(trigger, iterator, writer=writer,
                         lr_scheduler=lr_scheduler)

        # ToDo: remove the checkpoint_trigger, see pre_step
        self.checkpoint_trigger = IntervalTrigger.new(trigger)

        assert isinstance(metrics, dict) and metrics,  \
            'The metrics dict must not be empty!'
        self._checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.metrics = self._convert_metrics_to_internal_layout(metrics)
        self._keep_all = keep_all
        self.lr_scheduler = lr_scheduler
        if init_from_json:
            json_path = checkpoint_dir / self._json_filename
            assert checkpoint_dir.exists(), checkpoint_dir
            with json_path.open('r') as json_fd:
                json_state = json.load(json_fd)
            self.latest_checkpoint = (
                    self._checkpoint_dir / json_state['latest_checkpoint_path']
            )
            assert set(self.metrics.keys()) == set(json_state['metrics'].keys()), (
                self.metrics, json_state)
            for metric_key, metric in self.metrics.items():
                metric.set_state(json_state['metrics'][metric_key])
        else:
            if checkpoint_dir.exists():
                assert checkpoint_dir.is_dir(), checkpoint_dir
                assert len(list(checkpoint_dir.iterdir())) == 0, checkpoint_dir
            else:
                checkpoint_dir.mkdir()
            self.latest_checkpoint = None

    def set_last(self, iteration, epoch):
        super().set_last(iteration, epoch)
        self.checkpoint_trigger.set_last(iteration, epoch)

    def pre_step(self, trainer: 'pt.Trainer'):
        # A trigger triggers only once. So it is important to use here a copy
        # of self.trigger and super can use the original trigger.
        if self.checkpoint_trigger(
                iteration=trainer.iteration, epoch=trainer.epoch
        ):
            self._save_latest_checkpoint(trainer)
        super().pre_step(trainer)

    def dump_summary(self, trainer: 'pt.Trainer'):
        """ This class needs to overload the dump_summary - even if the naming
            is suboptimal - because the ValidationHook class produces the
            necessary metrics in its pre_step and immediately calls
            dump_summary. However, the implementation in SummaryHook clears
            the summary content.
        """
        self._update_validated_checkpoints()
        super().dump_summary(trainer)
        self._cleanup_stale_checkpoints()
        self.dump_json()

    def dump_json(self):
        """ Store the state information of the hok object to a json.
        """
        assert all(metric_key == metric.name
                   for metric_key, metric in self.metrics.items()), \
            'Some metric keys do not match their names!'
        json_path = self._checkpoint_dir / self._json_filename
        content = dict(
            latest_checkpoint_path=str(
                self.latest_checkpoint.relative_to(self._checkpoint_dir)
            ),
            metrics={metric_key: metric.to_json()
                     for metric_key, metric in self.metrics.items()})

        pb.io.dump_json(content, json_path)

    def close(self, trainer: 'pt.Trainer'):
        self._save_latest_checkpoint(trainer)
        self.dump_json()
        self.writer.close()

    @property
    def best_checkpoints(self):
        """ return a set of the checkpoints referenced as best by the metrics.
        """
        return {path
                for metric in self.metrics.values()
                for path in metric.paths}

    def _convert_metrics_to_internal_layout(self, metrics):
        return {metric_key: _Metric(metric_key, criterion,
                                    self._checkpoint_dir)
                for metric_key, criterion in metrics.items()}

    @property
    def latest_symlink_path(self):
        # ToDo why does resolve crash the test?
        # Resolve eliminates the symlink -> bad idea
        return (self._checkpoint_dir / f'ckpt_latest.{CKPT_EXT}').absolute()

    def _save_latest_checkpoint(self, trainer: 'pt.Trainer'):
        """ Unconditionally save a checkpoint for the current model.
            This is needed for resume of training.
        """
        checkpoint_path = trainer.default_checkpoint_path()

        trainer.save_checkpoint(checkpoint_path)
        self.latest_checkpoint = checkpoint_path

        # Create relative symlink to latest checkpoint
        if self.latest_symlink_path.is_symlink():
            self.latest_symlink_path.unlink()
        self.latest_symlink_path.symlink_to(checkpoint_path.name)

    def _update_validated_checkpoints(self):
        """ Save a checkpoint if the current model improves one or multiple
            validation metrics, dump the metric information to a json file
            and remove old checkpoints.
        """
        for metric_key, metric in self.metrics.items():
            if metric_key in self.summary['scalars']:
                summary_value = np.mean(self.summary['scalars'][metric_key])
            else:
                raise ValueError(metric_key, self.summary)

            if metric.is_better(summary_value):
                metric.update(summary_value, self.latest_checkpoint)

    def _cleanup_stale_checkpoints(self):
        """ Remove all checkpoints that became stale (i.e. have no associated
            metric where they perform best anymore).
        """
        if self._keep_all:
            return

        used_checkpoints \
            = set(self.best_checkpoints).union({self.latest_checkpoint})

        stored_checkpoints = [
            path for path in self._checkpoint_dir.glob(f'ckpt_*.{CKPT_EXT}')
            if path.is_file() and not path.is_symlink()]

        for checkpoint in stored_checkpoints:
            if checkpoint not in used_checkpoints:
                checkpoint.unlink()


class ProgressBarHook(TriggeredHook):
    """ Adds a progress bar to the console output. """
    def __init__(self, max_trigger, max_it_len=None, update_interval=10):
        """
        :param max_trigger: has to be defined if max_trigger unit is session
            integer with the length of the iterator
        :param max_it_len (int): length of iterator, only used if max_trigger
            uses unit epoch
        :param update_interval (int): Number of iterations to skip printing the
            progress bar.
        :param bar_length (int): Length of the progress bar in characters.
        :param disable: bool use to disable the entire progressbar wrapper
        """
        super().__init__((update_interval, 'iteration'))
        if isinstance(max_trigger, EndTrigger):
            length, unit = max_trigger.period, max_trigger.unit
        elif isinstance(max_trigger, (tuple, list)):
            length, unit = max_trigger
        else:
            raise ValueError(f'max_trigger is expected to be either a trigger '
                             f'or a list or tuple, but is {type(max_trigger)},'
                             f'{max_trigger}')
        if unit == 'iteration':
            max_iteration = length
        elif unit == 'epoch':
            if max_it_len is not None:
                max_iteration = length * max_it_len
            else:
                self.num_epochs = length
                max_iteration = progressbar.UnknownLength
        else:
            raise ValueError(f'unit {unit} is unknown,'
                             f' choose iteration or epoch')

        self.loss = None
        self.pbar = progressbar.ProgressBar(
            min_value=1,
            max_value=max_iteration,
            redirect_stderr=True,
            redirect_stdout=True,
            max_error=False,
        )

    @property
    def priority(self):
        return Priority.PROGRESS

    def set_last(self, iteration, epoch):
        super().set_last(iteration, epoch)
        self.pbar.value = iteration

    def pre_step(self, trainer: 'pt.Trainer'):
        iteration = trainer.iteration
        epoch = trainer.epoch
        if epoch == 1 and self.pbar.max_value is progressbar.UnknownLength:
            if hasattr(self, 'num_epochs'):
                # sets the max length of the bar after the first epoch
                self.pbar.max_value = (iteration +1) * self.num_epochs
        if self.trigger(iteration, epoch) and iteration > 1:
            self.pbar.update(iteration)

    def post_step(self, trainer: 'pt.Trainer', example,
                  model_output, review):
        self.loss = review["loss"]

    def close(self, trainer: 'pt.Trainer'):
        self.pbar.finish()


class StopTrainingHook(TriggeredHook):
    """ Raises a StopTraining exception if triggered. """
    def __init__(self, trigger):
        super().__init__(EndTrigger.new(trigger))

    @property
    def priority(self):
        return Priority.END

    def pre_step(self, trainer):
        if self.trigger(trainer.iteration, trainer.epoch):
            print(f'Training ended after {trainer.epoch} epochs and'
                  f' {trainer.iteration} iterations')
            raise StopTraining


class StopTraining(Exception):
    """ Rationale: Raised as signal to stop the training
        (e.g. when predefined number of iterations are completed.)
    """
    pass


class LossWeightAnnealingHook(TriggeredHook):
    """
    Anneals a loss weight within the los_weights dict of the trainer.
    """
    def __init__(self, name, factor, trigger, max_value=None, min_value=None):
        """

        Args:
            name: key of the loss_weight
            factor: factor by which to anneal the loss weight.
                factor > 1. results in an increase while factor < 1. results
                in a decrease
            trigger:
            max_value: upper bound of the weight
            min_value: lower bound of the weight
                (hint: can also be used to activate a loss weight after a
                certain number of iterations/epochs)
        """
        super().__init__(trigger)
        self.name = name
        self.factor = factor
        self.max_value = max_value
        self.min_value = min_value

    def pre_step(self, trainer):
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch) \
                and trainer.iteration != 0:
            weight = self.factor * trainer.loss_weights[self.name]
            if self.max_value is not None:
                weight = min(weight, self.max_value)
            if self.min_value is not None:
                weight = max(weight, self.min_value)
            trainer.loss_weights[self.name] = weight


class ModelAttributeAnnealingHook(TriggeredHook):
    """
    Anneals an attribute of the trainers model.
    """
    def __init__(
            self, name, trigger, factor=None, slope=None, max_value=None, min_value=None
    ):
        """

        Args:
            name: name of the attribute. You can use "attr1.attr11" to
                anneal a sub attribute
            factor: factor by which to anneal the attribute.
                factor > 1. results in an increase while factor < 1. results
                in a decrease
            trigger:
            max_value: upper bound of the weight
            min_value: lower bound of the weight
                (hint: can also be used to activate a loss weight after a
                certain number of iterations/epochs)
        """
        super().__init__(trigger)
        self.name = name.split('.')
        assert (factor is None) ^ (slope is None), (factor, slope)
        self.factor = factor
        self.slope = slope
        self.max_value = max_value
        self.min_value = min_value
        self.onset_value = None

    def get_module(self, trainer):
        module = trainer.model
        for attr_name in self.name[:-1]:
            module = getattr(module, attr_name)
        return module

    def pre_step(self, trainer):
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch) \
                and trainer.iteration != 0:
            module = self.get_module(trainer)
            value = getattr(module, self.name[-1])
            if self.onset_value is None:
                self.onset_value = value
            if self.factor is not None:
                value *= self.factor
            if self.slope is not None:
                value = self.onset_value + self.slope * trainer.iteration
            if self.max_value is not None:
                value = min(value, self.max_value)
            if self.min_value is not None:
                value = max(value, self.min_value)
            setattr(module, self.name[-1], value)

    def close(self, trainer):
        if self.onset_value is not None:
            module = self.get_module(trainer)
            setattr(module, self.name[-1], self.onset_value)
