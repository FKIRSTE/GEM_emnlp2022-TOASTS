import os
import pytest
import logging
from typing import Tuple

from pathlib import Path

from transformers import Seq2SeqTrainingArguments

from src.helpers.args import ModelArguments, DataTrainingArguments
from src.helpers.fn import (
    postprocess_text_bleu,
    postprocess_text_rouge_meteor,
    setup_logger,
    check_prefix,
    from_checkpoint,
    create_preprocess_function,
)

ArgType = Tuple[ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments]


@pytest.fixture()
def args() -> ArgType:

    data_args = DataTrainingArguments(
        dataset_name="dialogsum",
        max_eval_samples=1,
        max_train_samples=1,
        max_predict_samples=1,
    )
    model_args = ModelArguments(model_name_or_path="t5-small")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./output_tests",
        do_train=True,
        do_eval=True,
        do_predict=True,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        predict_with_generate=True,
    )

    return (model_args, data_args, training_args)


@pytest.fixture()
def logger() -> logging.Logger:

    logger = logging.getLogger(__name__)

    return logger


def test_from_checkpoint(logger: logging.Logger) -> None:

    output_dir = "./output"
    filename = "test_model.pt"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, filename)).touch()
    with pytest.raises(ValueError):
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir, do_train=True
        )
        from_checkpoint(training_args, logger)


def test_check_prefix(args: ArgType, logger: logging.Logger) -> None:
    check_prefix(args[1], args[0], logger)
