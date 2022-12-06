import pytest
import copy

from typing import Tuple, List

from transformers import Seq2SeqTrainingArguments

from src.finetuning import main
from src.helpers.args import ModelArguments, DataTrainingArguments


@pytest.fixture()
def standard_training_args() -> Seq2SeqTrainingArguments:
    training_args = Seq2SeqTrainingArguments(
        output_dir="./output_tests",
        # do_train=True, # github actions would fail due to memory consumption
        do_eval=True,
        do_predict=True,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        predict_with_generate=True,
        report_to="none",
    )

    return training_args


@pytest.fixture()
def data_args() -> List[DataTrainingArguments]:

    dataset_names = [
        "dialogsum",
        "qmsum_ami",
        "qmsum_icsi",
        "cnn_dailymail",
        "samsum",
        # "mediasum", # Agreed to remove mediasum (20.08.21 with Tirthankar, Terry, Norman, and Jan)
    ]
    data_args: List[DataTrainingArguments] = []

    for dataset_name in dataset_names:
        data_args.append(
            DataTrainingArguments(
                dataset_name=dataset_name,
                max_eval_samples=1,
                max_train_samples=1,
                max_predict_samples=1,
                ignore_pad_token_for_loss=True,
                pad_to_max_length=True,
                source_prefix="summarize: ",
            )
        )

    return data_args


@pytest.fixture()
def model_args() -> List[ModelArguments]:

    model_names = [
        "facebook/bart-base",
        # We remove these models to make the tests inexpensive
        # "t5-small",
        # "google/pegasus-large",
        # "google/byt5-base",
        # "facebook/bart-large-xsum",
        # "lrakotoson/scitldr-catts-xsum-ao",
        # These models are not listed for summarization in HF.
        # We need to either adapt or choose other models from Seq2SeqLM
        # "ctrl",
        # "hyunwoongko/ctrlsum-cnndm",
        # "EleutherAI/gpt-neo-1.3B",
        # "gpt2",
        # "nghuyong/ernie-2.0-en",
    ]

    return [
        ModelArguments(model_name_or_path=model_name, use_fast_tokenizer=False)
        for model_name in model_names
    ]


def test_main(
    model_args: List[ModelArguments],
    data_args: List[DataTrainingArguments],
    standard_training_args: Seq2SeqTrainingArguments,
) -> None:

    modified_training_args = copy.deepcopy(standard_training_args)
    modified_training_args.do_train = False
    modified_training_args.do_eval = False
    modified_training_args.do_predict = False
    with pytest.raises(ValueError):
        main(model_args[0], data_args[0], modified_training_args)

    modified_data_arg = copy.deepcopy(data_args[0])
    modified_data_arg.text_column = "topic"
    modified_data_arg.summary_column = "dialogue"
    main(model_args[0], modified_data_arg, standard_training_args)

    modified_training_args_2 = copy.deepcopy(standard_training_args)
    modified_training_args_2.label_smoothing_factor = 0.1
    main(model_args[0], data_args[0], modified_training_args_2)

    # Test all models on all datasets and all metrics
    for model_arg in model_args:
        for data_arg in data_args:
            main(model_arg, data_arg, standard_training_args)
