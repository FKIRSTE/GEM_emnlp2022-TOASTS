#!/usr/bin/env python
# coding=utf-8
# adaptation of run_summarization.py
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py

from typing import Dict, Union, Optional
import os
import logging
import wandb

import nltk  # Here to have a nice missing dependency error message early on

from datasets.dataset_dict import DatasetDict

from datasets import load_dataset
from filelock import FileLock
from transformers import (
    DataCollatorForSeq2Seq,
    set_seed,
    Seq2SeqTrainingArguments,
)

from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.file_utils import is_offline_mode
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import ray.tune as tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

from src.helpers.fn import (
    check_prefix,
    setup_logger,
    from_checkpoint,
    create_preprocess_function,
)
from src.models.models import get_model_fn_wrapper, get_tokenizer
from src.helpers.args import get_args, ModelArguments, DataTrainingArguments
from src.metrics.metrics import create_compute_metric_fn

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.2")

require_version("datasets>=1.8.0", "To fix: pipenv install.")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):  # pragma: no cover
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE"
            " first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


summarization_name_mapping = {
    "dialogsum": ("dialogue", "dialogue"),
    "cnn_dailymail": ("article", "highlights"),
    "qmsum_ami": ("transcript", "summary"),
    "qmsum_icsi": ("transcript", "summary"),
    "samsum": ("dialogue", "summary"),
    "mediasum": ("dialogue", "summary"),
    "ccdv/arxiv-summarization": ("article","abstract")
}

query_based_datasets = set(["qmsum_ami", "qmsum_icsi"])


def main(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: Seq2SeqTrainingArguments,
) -> Dict[str, float]:

    """[summary]

    Arguments:
        model_args {ModelArguments} -- [Arguments for the model]
        data_args {DataTrainingArguments} -- [Arguments for the dataset]
        training_args {Seq2SeqTrainingArguments} -- [Arguments for the training]

    Returns:
        A dictionary with the eval metrics
    """

    # Setup logger for all modules
    setup_logger(training_args=training_args, logger=logger)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device},"
        + f"n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)},"
        + f"16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check if prefix was given to avoid T5 error
    check_prefix(data_args=data_args, model_args=model_args, logger=logger)

    # Load the last checkpoint if possible
    last_checkpoint = from_checkpoint(training_args=training_args, logger=logger)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets
    data_loading_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data_prep",
        data_args.dataset_name + ".py",
    )
    raw_datasets: Optional[DatasetDict] = None
    if os.path.exists(data_loading_script):
        raw_datasets = load_dataset(
            data_loading_script,
            cache_dir=model_args.cache_dir,
        )  # type: ignore
    else:
        if "cnn_dailymail" in data_args.dataset_name:
            # this is for the test case only.
            # this is for the test case only.
            data_args.dataset_config_name = "3.0.0"
        if "reddit_tifu" in data_args.dataset_name:
            data_args.dataset_config_name = "long"
            data_args.columns_to_remove = ["ups", "num_comments", "upvote_ratio", "score"]
            data_args.split_input = [0.9, 0.05, 0.05]
        if "wikihow" in data_args.dataset_name:
            data_args.dataset_config_name = "all"
        if "scitldr" in data_args.dataset_name:
            data_args.columns_to_remove = ["source_labels", "rouge_scores", "paper_id"]
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            data_dir=data_args.dataset_dir,
            cache_dir=model_args.cache_dir,
            ignore_verifications=True
        )  # type: ignore

    if not raw_datasets:
        raise ValueError(
            "Dataset loading script not found locally or in huggingface/datasets library."
        )

    if len(data_args.columns_to_remove) > 0:
        raw_datasets = raw_datasets.remove_columns(data_args.columns_to_remove)

    if len(data_args.split_input) > 0:
        len_splits = len(data_args.split_input)

        if len_splits == 1:
            raise ValueError("[Split error]: Not enough splits given.")
        if len_splits > 3:
            raise ValueError("[Split error]: Too many splits given.")
        if sum(data_args.split_input) > 1:
            raise ValueError("[Split error]: The sum of split proportions exceeds 1.")
        if len_splits == 3 and sum(data_args.split_input) != 1:
            raise ValueError("[Split error]: The sum of the 3 splits does not add up to 1.")

        # The expected user input is sth like split(0.9, 0.05, 0.05), however we need split
        # like (0.9 | 0.5) meaning 0.9 of the inital dataset is for trianing and
        # the remaining 0.1 are splitted with 0.5. So an extra computation is required.
        train_testval_split = data_args.split_input[0]
        test_val_split = round(data_args.split_input[1] / (1 - train_testval_split), 5)  # convert
        logger.info(
            "[Split Info]: Splitting the dataset with internal splits "
            f"({train_testval_split} | {test_val_split})."
        )
        train_testval = raw_datasets["train"].train_test_split(train_size=train_testval_split)
        test_val = train_testval["test"].train_test_split(test_size=test_val_split)
        raw_datasets = DatasetDict(
            {
                "train": train_testval["train"],
                "test": test_val["test"],
                "validation": test_val["train"],
            }
        )

    tokenizer = get_tokenizer(model_args)
    get_model_fn = get_model_fn_wrapper(model_args, tokenizer)

    use_static_prefix: Optional[bool] = None
    prefix: Optional[str] = None
    if data_args.dataset_name not in query_based_datasets:
        prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    else:
        prefix = None
        use_static_prefix = False

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        raise ValueError(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}'"
                + f"needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' "
                + f"needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding: Union[str, bool] = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(
        model, "prepare_decoder_input_ids_from_labels"
    ):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels`"
            + f"method is not defined for `{model.__class__.__name__}`. This will lead to "
            + "loss being calculated twice and will take up more memory"
        )

    preprocess_function = create_preprocess_function(
        tokenizer,
        data_args,
        padding,
        max_target_length,
        text_column,
        summary_column,
        prefix,
        use_static_prefix,
        query_column="query",
    )

    train_dataset = None
    eval_dataset = None
    predict_dataset = None

    if training_args.do_train:  # pragma: no cover
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            if train_dataset:
                train_dataset = train_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )

    if training_args.do_eval:
        max_target_length = (
            data_args.val_max_target_length
            if data_args.val_max_target_length
            else data_args.max_target_length
        )
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            if eval_dataset:
                eval_dataset = eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )

    if training_args.do_predict:
        max_target_length = (
            data_args.val_max_target_length
            if data_args.val_max_target_length
            else data_args.max_target_length
        )
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            if predict_dataset:
                predict_dataset = predict_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        # model=get_model_fn, # Currently only works with models, not model_fn
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    compute_metrics = create_compute_metric_fn(
        tokenizer=tokenizer,
        data_args=data_args,
        evaluation_strategies=data_args.metrics,
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model_init=get_model_fn,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,  # type: ignore
        eval_dataset=eval_dataset if training_args.do_eval else None,  # type: ignore
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    print(training_args)

    # Hyperparameter sweep
    if data_args.hyperparameter_sweep:
        if not training_args.do_eval:
            raise ValueError("Hyperparameter sweep requires an evaluation to sweep for.")
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="ray",
            hp_space=lambda _: {
		"num_train_epochs": tune.choice([1,2,3,4,5]),
		"learning_rate": tune.uniform(1e-6, 5e-5),
		"weight_decay": tune.choice([0.05, 0.1, 0.2, 0.3, 0.4]),
		"per_device_train_batch_size": tune.choice([4,8]),
	    	"gradient_accumulation_steps": tune.choice([32,64])
	    },
	    # Choose among many libraries:
            # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
            search_alg=HyperOptSearch(metric="eval_rouge1", mode="max"),
            # Choose among schedulers:
            # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
            scheduler=ASHAScheduler(metric="eval_rouge1", mode="max"),
            # resources_per_trial={"cpu": 1, "gpu": 1},
            resources_per_trial={"cpu": 1, "gpu": 1},
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)  # type: ignore
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))  # type: ignore

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results: Dict[str, float] = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
            metric_key_prefix="eval",
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)  # type: ignore
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))  # type: ignore

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict and predict_dataset is not None:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,  # type: ignore
            metric_key_prefix="predict",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)  # type: ignore
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))  # type: ignore

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if training_args.predict_with_generate:
            predictions = tokenizer.batch_decode(
                predict_results.predictions,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = os.path.join(
                training_args.output_dir, "generated_predictions.txt"
            )
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(predictions))

    return results


if __name__ == "__main__":
    import os
    # See all possible arguments by passing the --help flag to this script.
    model_args, data_args, training_args = get_args()
    main(model_args, data_args, training_args)
