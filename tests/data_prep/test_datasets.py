import os
import pytest

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict


@pytest.fixture()
def data_loading_script_path() -> str:

    loader_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "src",
        "data_prep",
    )
    return loader_dir


def test_qmsum(data_loading_script_path: str) -> None:

    dataset_name_ami = "qmsum_ami"
    dataset_name_icsi = "qmsum_icsi"

    dataset_ami: DatasetDict = load_dataset(os.path.join(data_loading_script_path, dataset_name_ami + ".py"))  # type: ignore
    dataset_icsi: DatasetDict = load_dataset(os.path.join(data_loading_script_path, dataset_name_icsi + ".py"))  # type: ignore

    assert set(dataset_ami["train"].column_names) == set(["transcript", "summary", "query"])
    assert set(dataset_icsi["train"].column_names) == set(["transcript", "summary", "query"])
    assert dataset_ami["train"].column_names == dataset_ami["validation"].column_names
    assert dataset_ami["validation"].column_names == dataset_ami["test"].column_names
    assert dataset_icsi["train"].column_names == dataset_icsi["validation"].column_names
    assert dataset_icsi["validation"].column_names == dataset_icsi["test"].column_names


def test_dialogsum(data_loading_script_path: str) -> None:

    dataset_name = "dialogsum"
    dataset: DatasetDict = load_dataset(os.path.join(data_loading_script_path, dataset_name + ".py"))  # type: ignore

    assert set(dataset["train"].column_names) == set(["fname", "topic", "dialogue", "summary"])
    assert dataset["train"].column_names == dataset["validation"].column_names
    assert dataset["validation"].column_names == dataset["test"].column_names


def test_cnn_dailymail() -> None:

    dataset: DatasetDict = load_dataset("cnn_dailymail", "3.0.0")  # type: ignore

    assert set(dataset["train"].column_names) == set(["id", "article", "highlights"])
    assert dataset["train"].column_names == dataset["validation"].column_names
    assert dataset["validation"].column_names == dataset["test"].column_names


def test_samsum() -> None:

    dataset: DatasetDict = load_dataset("samsum")  # type: ignore

    assert set(dataset["train"].column_names) == set(["id", "dialogue", "summary"])
    assert dataset["train"].column_names == dataset["validation"].column_names
    assert dataset["validation"].column_names == dataset["test"].column_names
