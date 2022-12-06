import os
import sys

from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments, HfArgumentParser


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use one of the fast tokenizer "
                "(backed by the tokenizers library) or not."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": (
                "The specific model version to use (can be a branch name, tag name or commit id)."
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` "
                "(necessary to use this script with private models)."
            )
        },
    )
    training_style: str = field(
        default="random",
        metadata={
            "help": (
                "Set how the Trainer works for prefinetuning and prefintuning_oh."
                "Choose between: fix, random (default), CL_asc, CL_desc"
            )
        },
    )
    family_scaling: str = field(
        default="uniform",
        metadata={
            "help": (
                "Set how the Trainer works for prefinetuning and prefintuning_oh."
                "Choose between: uniform, proportional, temperature"
            )
        },
    )

    temperature: float = field(
        default=2.0,
        metadata={
            "help" : "Exponent T used for the temperature mixing. The mixing rate is raised by 1/T."
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    size_per_task: int = field(
        default=0,  # 14400,
        metadata={
            "help": "size of each dataset"
        },
    )

    num_tasks_per_family: int = field(
        default=3,
        metadata={
            "help": "Set the number of tasks per task family -> every family has the same number of tasks. Used to calculate the size_per_task and size_per_family."
        }
    )

    num_task_families: int = field(
        default=6,
        metadata={
            "help": "Specification how many task families are included in the training dataset. Used to calculate the size_per_task and size_per_family."
        }
    )

    num_stages: int = field(
        default=500,
        metadata={
            "help": "Number of stages during training. num_stages = total_num_batches_per_task / (num_task_families * num_tasks_per_family)"
        }
    )

    task_families_to_exclude: Optional[List[str]] = field(
        default_factory=lambda: [],
        metadata={"help": "short versions of task family names to exclude from pre-finetuning."}
    )

    dataset_name: str = field(
        default="dialogsum",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to dataset folder for manual data."},
    )
    columns_to_remove: Optional[List[str]] = field(
        default_factory=lambda: [],
        metadata={"help": "Columns to be removed from the dataset."},
    )
    split_input: Optional[List[float]] = field(
        default_factory=lambda: [],
        metadata={
            "help": (
                "Split the dataset according to the provided split proportions. "
                "The order is train, test, validation (optional)."
            )
        },
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the column in the datasets containing the full texts "
                "(for summarization)."
            )
        },
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the column in the datasets containing the summaries "
                "(for summarization)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    hyperparameter_sweep: bool = field(
        default=False,
        metadata={"help": "Perform a hyperparameter sweep."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pf_max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. "
                "Sequences longer than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pf_max_target_length: int = field(
        default=256,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. "
                "Sequences longer than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization."
                " Sequences longer than this will be truncated, sequences shorter will be padded."
                " Will default to `max_target_length`. This argument is also used to override the"
                " ``max_length`` param of ``model.generate``, which is used during ``evaluate``"
                " and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. If False, will pad"
                " the samples dynamically when batching to the maximum length in the batch. More"
                " efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training "
                "examples to this value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation "
                " examples to this value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of "
                "prediction examples to this value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to "
                "``model.generate``, which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to ignore the tokens corresponding to padded labels in the loss "
                "computation or not."
            )
        },
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "A prefix to add before every source text (useful for T5 models)."},
    )
    metrics: List[str] = field(
        default_factory=lambda: [
            "rouge",
            "meteor",
            "bertscore",  # There is a bug in the offical bertscore master. See issue #13.
            "bleu",
        ],
        metadata={"help": "Which metrics to evaluate summarization on. A list of strings."},
    )
    prefinetune: bool = field(default=True, metadata={"help": "Perform the pre-finetuning stage."})

    def __post_init__(self) -> None:
        if self.dataset_name is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def get_args() -> Tuple[
    ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments
]:  # pragma: no cover
    """Gets the command line arguments or arguments provided in a .json file if only the .json
    filepath is provided after calling the programdefined in the respective classes below.

    Returns:
        Tuple[ ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments ]: [Arguments]
    """
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args
