import pytest

from src.helpers.args import DataTrainingArguments


def test_has_dataset() -> None:

    model_args = DataTrainingArguments(
        dataset_name="dialogsum", max_target_length=512
    )
    assert model_args.val_max_target_length == 512
