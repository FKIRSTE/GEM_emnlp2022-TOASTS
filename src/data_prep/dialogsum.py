# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""DialogSum: A Real-life Scenario Dialogue Summarization Dataset"""


from typing import Dict, Generator, Any, Union, Optional, List, Tuple
import jsonlines

import datasets
from datasets.tasks import Summarization


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{Chen2021DialogSumAR,
  title={DialogSum: A Real-Life Scenario Dialogue Summarization Dataset},
  author={Yulong Chen and Yang Liu and Liang Chen and Yue Zhang},
  booktitle={FINDINGS},
  year={2021}
}
"""

_DESCRIPTION = """\
DialogSum is a real-life scenario dialogue summarization dataset, consisting of \
labeled dialogues with respective summaries. The dataset poses unique challenges \
in dialogue summarization, such as spoken terms, special discourse structures, \
coreferences and ellipsis, pragmatics and social common sense.
"""

_URL = "https://raw.githubusercontent.com/cylnlp/DialogSum/main/DialogSum_Data/"
_URLS = {
    "train": _URL + "dialogsum.train.jsonl",
    "validation": _URL + "dialogsum.dev.jsonl",
    "test": _URL + "dialogsum.test.jsonl",
}


class DialogSumConfig(datasets.BuilderConfig):
    """BuilderConfig for DialogSum."""

    def __init__(self, **kwargs):  # type: ignore
        """BuilderConfig for DialogSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


class DialogSum(datasets.GeneratorBasedBuilder):
    """DialogSum: A Real-life Scenario Dialogue Summarization Dataset"""

    BUILDER_CONFIGS = [
        DialogSumConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),  # type: ignore
            description="Plain text",
        ),
    ]

    def _info(self):  # type: ignore
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "fname": datasets.Value("string"),
                    "topic": datasets.Value("string"),
                    "dialogue": datasets.Value("string"),
                    "summary": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://github.com/cylnlp/DialogSum",
            citation=_CITATION,
            task_templates=[
                Summarization(text_column="dialogue", summary_column="summary")
            ],
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager  # type: ignore
    ) -> List[datasets.SplitGenerator]:
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=str(datasets.Split.TRAIN),
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=str(datasets.Split.VALIDATION),
                gen_kwargs={"filepath": downloaded_files["validation"]},
            ),
            datasets.SplitGenerator(
                name=str(datasets.Split.TEST),
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
        ]

    def _generate_examples(
        self, **kwargs: Union[str, Any]
    ) -> Generator[Tuple[int, Dict[str, str]], None, None]:
        """This function returns the examples in the raw (text) form."""

        key: int = 0
        filepath: Optional[str] = kwargs.get("filepath")

        logger.info("generating examples from = %s", filepath)
        with jsonlines.open(filepath, "r") as file:
            file_iterator: Generator[Dict[str, str], None, None] = file.iter()  # type: ignore
            for element in file_iterator:
                if "topic1" in element:
                    for counter in range(1, 3):
                        yield key, {
                            "fname": element["fname"],
                            "topic": element["topic" + str(counter)],
                            "dialogue": element["dialogue"],
                            "summary": element["summary" + str(counter)],
                        }
                        key += 1
                else:
                    yield key, {
                        "fname": element["fname"],
                        "topic": element["topic"],
                        "dialogue": element["dialogue"],
                        "summary": element["summary"],
                    }
                    key += 1
