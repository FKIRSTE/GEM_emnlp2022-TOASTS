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
"""MediaSum: A Large-scale Media Interview Dataset for Dialogue Summarization"""


from typing import Dict, Generator, Any, Union, Optional, List, Tuple
import os
import json

import datasets
from datasets.tasks import Summarization


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{zhu2021mediasum,
  title={MediaSum: A Large-scale Media Interview Dataset for Dialogue Summarization},
  author={Zhu, Chenguang and Liu, Yang and Mei, Jie and Zeng, Michael},
  journal={arXiv preprint arXiv:2103.06410},
  year={2021}
}
"""

_DESCRIPTION = """\
MediaSum, a large-scale media interview dataset consisting of 463.6K transcripts with abstractive
summaries. To create this dataset, we collect interview transcripts from NPR and CNN and employ the
overview and topic descriptions as summaries. Compared with existing public corpora for dialogue
summarization, our dataset is an order of magnitude larger and contains complex multi-party
conversations from multiple domains. We conduct statistical analysis to demonstrate the unique
positional bias exhibited in the transcripts of televised and radioed interviews. We also show that
MediaSum can be used in transfer learning to improve a model's performance on other dialogue
summarization tasks.
"""

_URL = "https://raw.githubusercontent.com/zcgzcgzcg1/MediaSum/main/data/"
_URLS = {
    "split_url": _URL + "train_val_test_split.json",
    "data": "https://drive.google.com/uc?export=download&id=1i8Ss_Lo5tseBirl99EnUjutN9gnU1iR8",
}


class MediaSumConfig(datasets.BuilderConfig):
    """BuilderConfig for DialogSum."""

    def __init__(self, **kwargs):  # type: ignore
        """BuilderConfig for DialogSum.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


class MediaSum(datasets.GeneratorBasedBuilder):
    """MediaSum: A Large-scale Media Interview Dataset for Dialogue Summarization"""

    BUILDER_CONFIGS = [
        MediaSumConfig(
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
                    "id": datasets.Value("string"),
                    "program": datasets.Value("string"),
                    "date": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "summary": datasets.Value("string"),
                    "utt": datasets.Sequence(feature=datasets.Value("string")),
                    "speaker": datasets.Sequence(feature=datasets.Value("string")),
                    "dialogue": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://github.com/zcgzcgzcg1/MediaSum",
            citation=_CITATION,
            task_templates=[Summarization(text_column="dialogue", summary_column="summary")],
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager  # type: ignore
    ) -> List[datasets.SplitGenerator]:
        downloaded_files = dl_manager.download_and_extract(_URLS)

        with open(os.path.join(downloaded_files["data"], "news_dialogue.json")) as json_file:
            data_dict = json.load(json_file)

            dataset_by_split: Dict[str, List[Dict[str, Union[str, List[str]]]]] = {
                "train": [],
                "val": [],
                "test": [],
            }

            for element in data_dict:
                if element["id"] in downloaded_files["split"]["train"]:
                    dataset_by_split["train"].append(element)
                elif element["id"] in downloaded_files["split"]["val"]:
                    dataset_by_split["val"].append(element)
                elif element["id"] in downloaded_files["test"]:
                    dataset_by_split["test"].append(element)

        return [
            datasets.SplitGenerator(
                name=str(datasets.Split.TRAIN),
                gen_kwargs={"elements": dataset_by_split["train"], "split": "train"},
            ),
            datasets.SplitGenerator(
                name=str(datasets.Split.VALIDATION),
                gen_kwargs={"elements": dataset_by_split["val"], "split": "val"},
            ),
            datasets.SplitGenerator(
                name=str(datasets.Split.TEST),
                gen_kwargs={"elements": dataset_by_split["test"], "split": "test"},
            ),
        ]

    def _generate_examples(
        self, **kwargs: Union[str, Any]
    ) -> Generator[Tuple[int, Dict[str, Union[List[str], str]]], None, None]:
        """This function returns the examples in the raw (text) form."""

        key: int = 0
        elements: Optional[List[Dict[str, Union[str, List[str]]]]] = kwargs.get(
            "elements"
        )  # type: ignore
        split: Optional[str] = kwargs.get("split")

        logger.info("generating examples from = %s", split)

        if elements:
            for element in elements:
                dialogue = ""
                for utt, speak in zip(element["utt"], element["speaker"]):
                    dialogue = dialogue + speak + " : " + utt + os.linesep
                print(dialogue)

                yield (
                    key,
                    {
                        "id": element["id"],
                        "program": element["program"],
                        "date": element["date"],
                        "url": element["url"],
                        "title": element["title"],
                        "summary": element["summary"],
                        "utt": element["utt"],
                        "speaker": element["speaker"],
                        "dialogue": dialogue,
                    },
                )
                key += 1
