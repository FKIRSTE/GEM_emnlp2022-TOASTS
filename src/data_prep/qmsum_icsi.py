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
"""QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization"""


from typing import Dict, Generator, Any, Union, Optional, List, Tuple
import os
import jsonlines

import datasets
from datasets.tasks import Summarization


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{zhong2021qmsum,
   title={{QMS}um: {A} {N}ew {B}enchmark for {Q}uery-based {M}ulti-domain {M}eeting {S}ummarization},
   author={Zhong, Ming and Yin, Da and Yu, Tao and Zaidi, Ahmad and Mutuma, Mutethia and Jha, Rahul and Hassan Awadallah, Ahmed and Celikyilmaz, Asli and Liu, Yang and Qiu, Xipeng and Radev, Dragomir},
   booktitle={North American Association for Computational Linguistics (NAACL)},
   year={2021}
}
"""

_DESCRIPTION = (
    "Meetings are a key component of human collaboration. As increasing numbers of meetings are"
    " recorded and transcribed, meeting summaries have become essential to remind those who may or"
    " may not have attended the meetings about the key decisions made and the tasks to be"
    " completed. However, it is hard to create a single short summary that covers all the content"
    " of a long meeting involving multiple people and topics. In order to satisfy the needs of"
    " different types of users, we define a new query-based multi-domain meeting summarization"
    " task, where models have to select and summarize relevant spans of meetings in response to a"
    " query, and we introduce QMSum, a new benchmark for this task. QMSum consists of 1,808"
    " query-summary pairs over 232 meetings in multiple domains. Besides, we investigate a"
    " locate-then-summarize method and evaluate a set of strong summarization baselines on the"
    " task. Experimental results and manual analysis reveal that QMSum presents significant"
    " challenges in long meeting summarization for future research."
)

_URL = "https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/Academic/jsonl/"
_URLS = {
    "train": _URL + "train.jsonl",
    "val": _URL + "val.jsonl",
    "test": _URL + "test.jsonl",
}


class QMSumAMIConfig(datasets.BuilderConfig):
    """BuilderConfig for QMSumAMI."""

    def __init__(self, **kwargs):  # type: ignore
        """BuilderConfig for QMSumAMI.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


class QMSumAMI(datasets.GeneratorBasedBuilder):
    """QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization"""

    BUILDER_CONFIGS = [
        QMSumAMIConfig(
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
                    "transcript": datasets.Value("string"),
                    "query": datasets.Value("string"),
                    "summary": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://github.com/Yale-LILY/QMSum",
            citation=_CITATION,
            task_templates=[Summarization(text_column="transcript", summary_column="summary")],
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
                gen_kwargs={"filepath": downloaded_files["val"]},
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
            file_iterator: Generator[
                Dict[str, List[Dict[str, str]]], None, None
            ] = file.iter()  # type: ignore
            for element in file_iterator:
                transcript: str = ""
                for partial in element["meeting_transcripts"]:
                    transcript = (
                        transcript + partial["speaker"] + " : " + partial["content"] + os.linesep
                    )
                if len(transcript) == 0:
                    raise ValueError("No transcript found.")

                for query_and_summary in element["general_query_list"]:

                    yield key, {
                        "transcript": transcript,
                        "query": query_and_summary["query"],
                        "summary": query_and_summary["answer"],
                    }
                    key += 1
