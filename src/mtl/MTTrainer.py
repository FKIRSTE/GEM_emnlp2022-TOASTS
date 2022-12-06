import transformers
import torch
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DataCollator, InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict
import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
from dataclasses import dataclass

import torch.utils.data as data_utils

from src.helpers.args import get_args, ModelArguments, DataTrainingArguments

#class MTDataCollator:  # (DataCollator):  # ToDo: Why do we not use the DataCollatorForSeq2Seq? -> we already tokenized/padded
"""
Extending the existing DataCollator to work with NLP dataset batches.
"""

train_dataset_error_literals = "Trainer: training requires a train_dataset."

def MTDataCollator(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
    if isinstance(features[0], dict):
        first = features[0]
        batch = {}

        if "labels" in first and first["labels"] is not None:
            _add_label_to_batch(features, first, batch)

        elif "label_ids" in first and first["label_ids"] is not None:
            print("LABEL IDs")

        for k, v in first.items():
            _add_features_to_batch(features, batch, k, v)
        return batch
    else:
        # otherwise, revert to using the default collate_batch
        return transformers.data.data_collator.default_data_collator(features)

def _add_features_to_batch(features, batch, k, v):
    if k not in ("labels", "label_ids") and v is not None and not isinstance(v, str):
        if isinstance(v, torch.Tensor):
            batch[k] = torch.stack([f[k] for f in features])
        else:
            batch[k] = torch.tensor([f[k] for f in features])

def _add_label_to_batch(features, first, batch):
    label = first["labels"].item() if isinstance(first["labels"], torch.Tensor) else first["labels"]
    dtype = torch.long if isinstance(label, int) else torch.float
    batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=dtype)

@dataclass
class WinograndeDataCollator:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "answer" if "answer" in features[0].keys() else "answers"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["answer"] = torch.tensor(labels, dtype=torch.int64)
        return batch

class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """
    def to(self, device):
        return self


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """
    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch
class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """
    def __init__(self, dataloader_dict, steps_delta, batch_order='random', num_batches_dict=None):
        self.dataloader_dict = dataloader_dict

        self.NUM_TASKS = len(self.dataloader_dict)
        self.STEPS_DELTA = steps_delta #118 #90 #133  # 68  = size(smallest_ds)//(self.NUM_TASKS * BATCH_SIZE)
        self.NUM_STEPS = (self.NUM_TASKS) * self.STEPS_DELTA

        self.num_batches_dict = num_batches_dict
        if self.num_batches_dict == None:
            self.num_batches_dict = {
                task_name: self.NUM_STEPS #len(dataloader)   #Set this to our value and not the dataset length. We want a total of 50k steps per task.
                for task_name, _ in self.dataloader_dict.items()
            }  # ==NUM_BATCHES per TASK_NAME
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset)
            for dataloader in self.dataloader_dict.values()
        )

        self.batch_order = batch_order.lower()
        self.current_task_name = ''

    def __len__(self):
        return sum(self.num_batches_dict.values())  # ==NUM_TASKS

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """

        CL_ORDER_DESC = True
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)

        if 'cl' in self.batch_order:
            task_loop_list = []
            CL_ORDER_DESC = self.batch_order == 'cl_desc'

            task_loop_list = self._cl_setup(CL_ORDER_DESC, task_choice_list, task_loop_list)
            task_choice_list=task_loop_list

        elif self.batch_order == 'random':
            np.random.shuffle(task_choice_list)

        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }

        try:
            for task_choice in task_choice_list:
                task_name = self.task_name_list[task_choice]
                self.current_task_name = task_name
                yield next(dataloader_iter_dict[task_name])
        except StopIteration:
            print(f"out of data #{self.current_task_name}")

    def _cl_setup(self, cl_order_desc, task_choice_list, task_loop_list):
        for stage in range(1, self.NUM_TASKS+1):
            tasks_range = []
            if cl_order_desc:
                print("CL DESC ORDER")
                tasks_range = list(range(stage-1, -1, -1))  # CL order v1 1-21-321-4321-
            else:
                print("CL ASC ORDER")
                tasks_range = [stage-1] + list(range(0, stage-1, 1))  # CL order v2 1-21-312-4123

            for i in tasks_range:
                j=0 if (i==stage-1) else stage-1
                task_pos = i*self.NUM_STEPS
                seq_start = j*self.STEPS_DELTA + task_pos
                seq_end = stage*self.STEPS_DELTA + task_pos
                task_loop_list += list(task_choice_list[seq_start : seq_end])

        return task_loop_list

class MultitaskTrainer_OH(transformers.Trainer):
    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError(train_dataset_error_literals)

        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader =DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
        )
        return data_loader


    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError(train_dataset_error_literals)
        train_dataset = self.train_dataset


        min_list = [len(v) for v in train_dataset.values()]
        num_task = len(min_list)
        min_size = min(min_list)  # should be equal to max_size from earlier
        steps_delta = min_size//(num_task * self.args.train_batch_size)
        steps_delta = max(steps_delta, 1)
        print(f"NUM: {num_task} || MIN SIZE: {min_size} || STEPS DELTA: {steps_delta}")

        model_args, _, _ = get_args()
        num_batches_dict = None if model_args.family_scaling == "uniform" else {t : int(len(v) / self.args.train_batch_size) for t, v in self.train_dataset.items()}


        return MultitaskDataloader({
            task_name: self.get_single_train_dataloader(task_name, task_dataset)
            for task_name, task_dataset in train_dataset.items()
        },steps_delta,batch_order=model_args.training_style, num_batches_dict=num_batches_dict)

class HeterogeneousTrainer(transformers.Trainer):

    def get_train_dataloader(self):
        """
        """

        if self.train_dataset is None:
            raise ValueError(train_dataset_error_literals)

        train_dataset = self.train_dataset

        dataset = []
        for key in train_dataset.keys():
            dataset.append(train_dataset[key])

        dataset = data_utils.ConcatDataset(dataset)
        loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
        )

        return loader

    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError(train_dataset_error_literals)

        train_sampler = (
            RandomSampler(train_dataset)  # ToDo: change for heterogeneous batches
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader =DataLoader(
              train_dataset,
              batch_size=self.args.train_batch_size,
              sampler=train_sampler,
              collate_fn=self.data_collator,
        )
        return data_loader


    def get_train_dataloader(self):
        import torch.utils.data as data_utils

        if self.train_dataset is None:
            raise ValueError(train_dataset_error_literals)
        train_dataset = self.train_dataset


        task_family_dict = {
            'go_emotions': 'cls',
            'imdb': 'cls',
            'ag_news': 'cls',
            'xsum': 'sum',
            'wiki_lingua': 'sum',
            'aeslc': 'sum',
            'boolq': 'rc',
            'squad': 'rc',
            'tweet_qa': 'rc',
            'hotpot_qa': 'cbqa',
            'nq': 'cbqa',
            'record': 'cbqa',
            'winogrande': 'cmns',
            'piqa': 'cmns',
            'socqa':'cmns',
            'mnli': 'nli',
            'anli': 'nli',
            'qnli': 'nli',
        }

        family_dataset = {}
        for task_name, task_dataset in train_dataset.items():
            key = task_family_dict[task_name]
            if key not in family_dataset.keys():
                family_dataset[key] = []
            current_ds = family_dataset[key]
            current_ds.append(task_dataset)
            family_dataset[key] = current_ds

        for key in family_dataset.keys():
            family_dataset[key] = data_utils.ConcatDataset(family_dataset[key])

        print(f"\n\nFAMILY DS: {family_dataset}\n\n")
        min_list = [len(v) for v in family_dataset.values()]

        num_task = len(min_list)
        min_size = min(min_list)  # should be equal to max_size from earlier
        steps_delta = min_size//(num_task * self.args.train_batch_size)
        steps_delta = max(steps_delta, 1)
        print(f"NUM: {num_task} || MIN SIZE: {min_size} || STEPS DELTA: {steps_delta}")

        model_args, _, _ = get_args()

        return MultitaskDataloader({
            task_name: self.get_single_train_dataloader(task_name, task_dataset)
            for task_name, task_dataset in family_dataset.items()
        },steps_delta,batch_order=model_args.training_style,)