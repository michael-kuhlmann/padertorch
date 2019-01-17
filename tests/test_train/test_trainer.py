import tempfile
from pathlib import Path
import contextlib
import inspect
import textwrap
from unittest import mock

import numpy as np
import torch

import tensorflow as tf
from google.protobuf.json_format import MessageToDict

import padertorch as pt
import paderbox as pb


class Model(pt.Model):

    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(28 * 28, 10)

    def forward(self, inputs):
        clean = inputs['image']

        if isinstance(clean, np.ndarray):
            clean = torch.tensor(clean)

        image = torch.reshape(clean, [-1])

        return self.l(image)

    def review(self, inputs, output):
        digits = inputs['digit']

        target = torch.tensor(
            np.array(digits).astype(np.int64),
            device=output.device,
        )[None]
        ce = torch.nn.CrossEntropyLoss()(output[None, :], target)
        return {'losses': {'loss': ce}}


def get_iterators():
    db = pb.database.mnist.MnistDatabase()
    return (
        db.get_iterator_by_names('train'),
        db.get_iterator_by_names('test'),
    )


class TriggerMock(pt.train.trigger.Trigger):
    def __init__(self, trigger, log_list):
        self.trigger = trigger
        self.log_list = log_list

    def __call__(self, iteration, epoch):
        ret = self.trigger(iteration, epoch)

        frame = inspect.stack()[1].frame

        if 'self' in frame.f_locals:
            name = frame.f_locals['self'].__class__.__name__

            string = f'I:{iteration}, E: {epoch}, {ret}, {name}.{inspect.stack()[1].function}'
            self.log_list.append(
                string
            )

            # print(string)
        else:
            callerframerecord = inspect.stack()[2]
            frame = callerframerecord[0]
            # if 'self' in frame.f_locals:
            name = frame.f_locals['self'].__class__.__name__
            assert name == 'OrTrigger', name

        return ret

    def set_last(self, iteration, epoch):
        return self.trigger.set_last(iteration=iteration, epoch=epoch)


@contextlib.contextmanager
def record_hook_trigger_calls(trainer):

    class GetDefaultHooksMock(mock.MagicMock):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.log_list = []

        def __call__(self, *args, **kw):
            print('+'*80)
            print('args, kw', args, kw)
            print('+'*40)
            hooks = super().__call__(*args, **kw)

            for hook in hooks:
                for k, v in list(hook.__dict__.items()):
                    if isinstance(v, pt.train.trigger.Trigger):
                        hook.__dict__[k] = TriggerMock(v, self.log_list)

            return hooks

    with mock.patch.object(
            trainer,
            'get_default_hooks',
            wraps=trainer.get_default_hooks,
            new_callable=GetDefaultHooksMock,
    ) as mocked:
        yield mocked


def load_tfevents_as_dict(
        path
):
    """

    >>> path = '/net/home/boeddeker/sacred/torch/am/32/events.out.tfevents.1545605113.ntsim1'
    >>> load_tfevents_as_dict(path)[2]
    {'wall_time': 1545605119.7274427, 'step': 1, 'summary': {'value': [{'tag': 'training/grad_norm', 'simple_value': 0.21423661708831787}]}}

    """
    # MessageToDict(e, preserving_proto_field_name=True)
    #   Converts int to str
    return [
        MessageToDict(e)
        for e in tf.train.summary_iterator(str(path))
    ]


def test_single_model():
    it_tr, it_dt = get_iterators()
    it_tr = it_tr[:2]
    it_dt = it_dt[:2]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        config = pt.Trainer.get_config(
            updates=pb.utils.nested.deflatten({
                'model.cls': Model,
                'storage_dir': str(tmp_dir),
                'max_trigger': (2, 'epoch'),
                'keep_all_checkpoints': True,
            })
        )

        t = pt.Trainer.from_config(config)

        files_before = tuple(tmp_dir.glob('*'))
        if len(files_before) != 0:
            # no event file
            raise Exception(files_before)

        with record_hook_trigger_calls(t) as mocked:
            t.train(
                train_iterator=it_tr,
                validation_iterator=it_dt,
                hooks=None,
                metrics={'loss': 'min'},
                n_best_checkpoints=1,
            )

        hook_calls = ('\n'.join(mocked.log_list))


        # CheckpointedValidationHook trigger is called two times
        #   (once for checkpointing once for validation)

        hook_calls_ref = textwrap.dedent('''
        I:0, E: 0, False, SummaryHook.pre_step
        I:0, E: 0, False, CheckpointedValidationHook.pre_step
        I:0, E: 0, False, CheckpointedValidationHook.pre_step
        I:0, E: 0, False, StopTrainingHook.pre_step
        I:0, E: 0, False, ProgressBarHook.post_step
        I:0, E: 0, False, ProgressBarHook.post_step
        I:1, E: 0, False, SummaryHook.pre_step
        I:1, E: 0, False, CheckpointedValidationHook.pre_step
        I:1, E: 0, False, CheckpointedValidationHook.pre_step
        I:1, E: 0, False, StopTrainingHook.pre_step
        I:1, E: 0, True, ProgressBarHook.post_step
        I:1, E: 0, False, ProgressBarHook.post_step
        I:2, E: 0, False, SummaryHook.pre_step
        I:2, E: 0, False, CheckpointedValidationHook.pre_step
        I:2, E: 0, False, CheckpointedValidationHook.pre_step
        I:2, E: 0, False, StopTrainingHook.pre_step
        I:2, E: 1, True, SummaryHook.pre_step
        I:2, E: 1, True, CheckpointedValidationHook.pre_step
        I:2, E: 1, True, CheckpointedValidationHook.pre_step
        I:2, E: 1, False, StopTrainingHook.pre_step
        I:2, E: 1, True, ProgressBarHook.post_step
        I:2, E: 1, True, ProgressBarHook.post_step
        I:3, E: 1, False, SummaryHook.pre_step
        I:3, E: 1, False, CheckpointedValidationHook.pre_step
        I:3, E: 1, False, CheckpointedValidationHook.pre_step
        I:3, E: 1, False, StopTrainingHook.pre_step
        I:3, E: 1, True, ProgressBarHook.post_step
        I:3, E: 1, False, ProgressBarHook.post_step
        I:4, E: 1, False, SummaryHook.pre_step
        I:4, E: 1, False, CheckpointedValidationHook.pre_step
        I:4, E: 1, False, CheckpointedValidationHook.pre_step
        I:4, E: 1, False, StopTrainingHook.pre_step
        I:4, E: 2, True, SummaryHook.pre_step
        I:4, E: 2, True, CheckpointedValidationHook.pre_step
        I:4, E: 2, True, CheckpointedValidationHook.pre_step
        I:4, E: 2, True, StopTrainingHook.pre_step
        ''').strip()

        print('#' * 80)
        print(hook_calls)
        print('#' * 80)

        assert hook_calls == hook_calls_ref, (hook_calls == hook_calls_ref)

        files_after = tuple(tmp_dir.glob('*'))
        assert len(files_after) == 2, files_after
        for file in files_after:
            if 'tfevents' in file.name:
                events = load_tfevents_as_dict(file)
                assert len(events) == 19, (len(events), events)

                tags = []
                for event in events:
                    if 'summary' in event.keys():
                        value, = event['summary']['value']
                        tags.append(value['tag'])

                import itertools, collections
                c = dict(collections.Counter(tags))
                expect = {
                    'training/grad_norm': 3,
                    'training/grad_norm_': 3,
                    'training/loss': 3,
                    'training/time_per_step': 3,
                    'training/time_rel_data_loading': 3,
                    'training/time_rel_train_step': 3,
                }
                assert c == expect, c

            elif file.name == 'checkpoints':
                checkpoints_files = tuple(file.glob('*'))
                assert len(checkpoints_files) == 4, checkpoints_files
                checkpoints_files_name = [
                    f.name
                    for f in checkpoints_files
                ]
                expect = {
                    'ckpt_1.pth', 'ckpt_2.pth', 'ckpt_4.pth', 'ckpt_state.json'
                }
                assert expect == set(checkpoints_files_name), (
                    expect, checkpoints_files_name
                )
            else:
                raise ValueError(file)

        config['kwargs']['max_trigger'] = (4, 'epoch')
        config['kwargs']['init_checkpoint'] = tmp_dir / 'checkpoints' / 'ckpt_4.pth'
        t = pt.Trainer.from_config(config)

        with record_hook_trigger_calls(t) as mocked:
            t.train(
                train_iterator=it_tr,
                validation_iterator=it_dt,
                hooks=None,
                metrics={'loss': 'min'},
                n_best_checkpoints=1,
            )


        hook_calls = ('\n'.join(mocked.log_list))

        hook_calls_ref = textwrap.dedent('''
        I:4, E: 2, False, SummaryHook.pre_step
        I:4, E: 2, False, CheckpointedValidationHook.pre_step
        I:4, E: 2, False, CheckpointedValidationHook.pre_step
        I:4, E: 2, False, StopTrainingHook.pre_step
        I:4, E: 2, False, ProgressBarHook.post_step
        I:4, E: 2, False, ProgressBarHook.post_step
        I:5, E: 2, False, SummaryHook.pre_step
        I:5, E: 2, False, CheckpointedValidationHook.pre_step
        I:5, E: 2, False, CheckpointedValidationHook.pre_step
        I:5, E: 2, False, StopTrainingHook.pre_step
        I:5, E: 2, True, ProgressBarHook.post_step
        I:5, E: 2, False, ProgressBarHook.post_step
        I:6, E: 2, False, SummaryHook.pre_step
        I:6, E: 2, False, CheckpointedValidationHook.pre_step
        I:6, E: 2, False, CheckpointedValidationHook.pre_step
        I:6, E: 2, False, StopTrainingHook.pre_step
        I:6, E: 3, True, SummaryHook.pre_step
        I:6, E: 3, True, CheckpointedValidationHook.pre_step
        I:6, E: 3, True, CheckpointedValidationHook.pre_step
        I:6, E: 3, False, StopTrainingHook.pre_step
        I:6, E: 3, True, ProgressBarHook.post_step
        I:6, E: 3, True, ProgressBarHook.post_step
        I:7, E: 3, False, SummaryHook.pre_step
        I:7, E: 3, False, CheckpointedValidationHook.pre_step
        I:7, E: 3, False, CheckpointedValidationHook.pre_step
        I:7, E: 3, False, StopTrainingHook.pre_step
        I:7, E: 3, True, ProgressBarHook.post_step
        I:7, E: 3, False, ProgressBarHook.post_step
        I:8, E: 3, False, SummaryHook.pre_step
        I:8, E: 3, False, CheckpointedValidationHook.pre_step
        I:8, E: 3, False, CheckpointedValidationHook.pre_step
        I:8, E: 3, False, StopTrainingHook.pre_step
        I:8, E: 4, True, SummaryHook.pre_step
        I:8, E: 4, True, CheckpointedValidationHook.pre_step
        I:8, E: 4, True, CheckpointedValidationHook.pre_step
        I:8, E: 4, True, StopTrainingHook.pre_step
        ''').strip()

        print('#' * 80)
        print(hook_calls)
        print('#' * 80)

        assert hook_calls == hook_calls_ref, (hook_calls == hook_calls_ref)

        files_after = tuple(tmp_dir.glob('*'))
        assert len(files_after) == 2, files_after
        for file in files_after:
            if 'tfevents' in file.name:
                file_bytes = file.read_bytes()
                # 913
                assert len(file_bytes) > 800, len(file_bytes)
                assert len(file_bytes) < 1000, len(file_bytes)
            elif file.name == 'checkpoints':
                checkpoints_files = tuple(file.glob('*'))
                assert len(checkpoints_files) == 6, checkpoints_files
                checkpoints_files_name = [
                    f.name
                    for f in checkpoints_files
                ]
                expect = {
                    *[f'ckpt_{i}.pth'for i in [1, 2, 4, 6, 8]],
                    'ckpt_state.json',
                }
                assert expect == set(checkpoints_files_name), (
                    expect, checkpoints_files_name
                )
            else:
                raise ValueError(file)