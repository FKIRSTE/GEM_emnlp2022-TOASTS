from typing import Union, List, Tuple, Any, Dict, Callable, Optional

import os
import sys
import logging
import nltk
import transformers
import datasets

from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    Seq2SeqTrainingArguments,
    PreTrainedTokenizer,
    BatchEncoding,
)

from src.helpers.args import DataTrainingArguments, ModelArguments


def setup_logger(
    training_args: Seq2SeqTrainingArguments, logger: logging.Logger
) -> None:  # pragma: no cover
    """Setups all loggers for printing purposes

    Args:
        training_args (Seq2SeqTrainingArguments): [Training Arguments]
        logger (logging.Logger): [Initiated logger from logging]
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def check_prefix(
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    logger: logging.Logger,
) -> None:
    """Checks whether a source_prefix was given when specific models are used.

    Args:
        data_args (DataTrainingArguments): [DataTrainingArguments]
        model_args (ModelArguments): [ModelArguments]
        logger (logging.Logger): [The standard logger initiated from logging.]
    """
    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, "
            "e.g. with `--source_prefix 'summarize: ' `"
        )


def from_checkpoint(
    training_args: Seq2SeqTrainingArguments, logger: logging.Logger
) -> Union[None, str]:
    """Initiates a model from a local checkpoint

    Args:
        training_args (Seq2SeqTrainingArguments): [Seq2SeqTrainingArguments]
        logger (logging.Logger): [The standard logger initiated from logging.]

    Raises:
        ValueError: [When the output directory is not empty and overwriting is not speicified.]

    Returns:
        Union[None, str]: [The last checkpoint if it exists.]
    """

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        if (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):  # pragma: no cover
            # We would need a dummy checkpoint to test.
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train"
                " from scratch."
            )
    return last_checkpoint


def postprocess_text_rouge_meteor(
    preds: List[str], labels: List[str]
) -> Tuple[List[str], List[str]]:
    """Preprocesses the predictions and labels in the rouge and meteor format.
    This format expects a list of sentence with newline after each sentence.

    Args:
        preds (List[str]): [The list of predicted sentences.]
        labels (List[str]): [The list of actual target sentences.]

    Returns:
        Tuple[List[str], List[str]]: [Returns a new list of sentences according to expected format]
    """

    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rouge and meteor expects a list of sentences with newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def postprocess_text_bleu(
    preds: List[str], labels: List[str]
) -> Tuple[List[List[str]], List[List[List[str]]]]:
    """Preprocesses the predictions and labels in the blue.
    This format expects a list of tokens as predictions and a list of candidates each containing
    tokens as labels


    Args:
        preds (List[str]): [The list of predicted sentences.]
        labels (List[str]): [The list of actual target sentences.]

    Returns:
        Tuple[List[List[str]], List[List[List[str]]]]: [Returns the expected structure.]
    """
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # bleu expects a list of tokens as predictions and a list of candidates each containing
    # tokens as labels
    final_predictions: List[List[str]] = [nltk.word_tokenize(pred) for pred in preds]
    final_labels: List[List[List[str]]] = [[nltk.word_tokenize(label)] for label in labels]

    return final_predictions, final_labels


def create_preprocess_function(
    tokenizer: PreTrainedTokenizer,
    data_args: DataTrainingArguments,
    padding: Union[str, bool],
    max_target_length: int,
    text_column: str,
    summary_column: str,
    prefix: Optional[str],
    use_static_prefix: Optional[bool],
    query_column: Optional[str],
) -> Callable[[Dict[str, Any]], BatchEncoding]:
    """Creates the preprocess function for models.

    Args:
        tokenizer (PreTrainedTokenizer): [PreTrainedTokenizer]
        data_args (DataTrainingArguments): [DataTrainingArguments]
        padding (Union[str, bool]): [Whether to use padding (bool) of which strategy to use (str)]
        max_target_length (int): [The max length of tokens]
        text_column (str): [Which column to use for the source sentence.]
        summary_column (str): [Which column to use for the target summary.]
        prefix (str): [Which prefix to give the model as indicative for summarization.]

    Returns:
        Callable[[Dict[str, Any]], Dict[str, torch.Tensor]]: [description]
    """

    def preprocess_function(examples: Dict[str, Any]) -> BatchEncoding:
        inputs = examples[text_column]
        targets = examples[summary_column]

        print(use_static_prefix, query_column, prefix)
        if use_static_prefix is not None and query_column is not None and prefix is None:
            queries = examples[query_column]
            inputs = [query + inp for query, inp in zip(inputs, queries)]
        else:
            if prefix is not None:
                inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )
        """The actual preprocess function called by the Trainer. Uses the outer scope variables to
        modify the examples.

        Returns:
            [BatchEncoding]: [A dict of input names (str) abd tokenized inputs
            (torch.Tensor)]
        """
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_length,
                padding=padding,
                truncation=True,
            )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when
        # we want to ignore padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:

            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]  # type: ignore
            ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return preprocess_function
