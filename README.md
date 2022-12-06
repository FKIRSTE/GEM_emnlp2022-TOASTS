# Analyzing Multi-Task Learning for Abstractive Text Summarization

[![arXiv](https://img.shields.io/badge/arXiv-2210.14606-red.svg)](https://arxiv.org/abs/2210.14606)

## Abstract
Despite the recent success of multi-task learning and pre-finetuning for natural language understanding, few works have studied the effects of task families on abstractive text summarization. Task families are a form of task grouping during the pre-finetuning stage to learn common skills, such as reading comprehension. To close this gap, we analyze the influence of multi-task learning strategies using task families for the English abstractive text summarization task. We group tasks into one of three strategies, i.e., sequential, simultaneous, and continual multi-task learning, and evaluate trained models through two downstream tasks. We find that certain combinations of task families (e.g., advanced reading comprehension and natural language inference) positively impact downstream performance. Further, we find that choice and combinations of task families influence downstream performance more than the training scheme, supporting the use of task families for abstractive text summarization.

## Results
Our discussed [results](https://www.gipp.com/wp-content/papercite-data/pdf/kirstein2022a.pdf#page=6 "discussed results") are shown in table 3-6.
All gathered [results](https://www.gipp.com/wp-content/papercite-data/pdf/kirstein2022a.pdf#page=16 "all results") are in table 11-40.

## Quick Start

### Install

```bash
pip install pipenv

# Clone this repo and change directory
git clone git@github.com:FKIRSTE/GEM_emnlp2022-TOASTS.git
cd SummER

# Create python environment
pip install pipenv
pipenv install --dev
```

### Run

> Pre-finetuning comes with two options: ```src.mtl.prefinetuning_oh``` for sequential training and continuous multi-task learning training, ```src.mtl.prefinetuning_hb``` for simultaneous training.
```bash
# Pre-finetuning
pipenv run python -m src.mtl.prefinetuning_oh --model_name_or_path facebook/bart-large --output_dir prefinetuned_oh --overwrite_output_dir --pf_max_source_length 512 --pf_max_target_length 128 --num_stages 500 --auto_find_batch_size --fp16

# Finetuning
pipenv run python -m src.finetuning --model_name_or_path ./prefinetuned_oh --dataset_name ccdv/arxiv-summarization --output_dir finetuning_oh --save_total_limit 2 --save_strategy epoch --do_train --do_eval --do_predict --predict_with_generate --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --fp16
```

Available options for pre-finetuning:
```bash
'--task-families_to_exclude': (Optional[List[str]]) use short versions of the task family names (i.e., cbqs, cls, cmns, nli, rc, sum) to exclude them from the trainining process. cbqs is the internal name of RC+.
'--training_style': (str) Set the training scheme. Available options are fix, random, CL_asc.
'--family_scaling': (str) Set the intra-family scaling scheme. Available options are uniform, proportional.
```
Available options for finetuning:
```bash
'--dataset_name': (str) The name of the dataset to use via the datasets library. For the paper, we used ccdv/arxiv-summarization; reddit_tifu
```


For help, run the following command:

> For pre-finetuning, run one of these two:
```bash
pipenv run python -m src.mtl.prefinetuning_oh --help
pipenv run python -m src.mtl.prefinetuning_hb --help
```
> For finetuning, run :
```bash
pipenv run python -m src.finetuning --help
```

## Citation
```bib
@inproceedings{Kirstein2022a,
	title        = {Analyzing Multi-Task Learning for Abstractive Text Summarization},
	author       = {Kirstein, Frederic and Wahle, Jan Philipp and Ruas, Terry and Gipp, Bela},
	year         = 2022,
	month        = {Dec.},
	booktitle    = {Proceedings of the 2nd Workshop on Natural Language Generation, Evaluation, and Metrics (GEM 2022)},
	location     = {Abu Dhabi, United Arab Emirates},
	publisher    = {Association for Computational Linguistics},
	topic        = {nlp}
}
```
## License
This repository is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
Use the code for any of your research projects, but be nice and give credit where credit is due.
Any illegal use for plagiarism or other purposes is prohibited.