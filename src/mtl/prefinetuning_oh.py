from typing import Union
import transformers
from transformers import (
    set_seed,
    Seq2SeqTrainingArguments,
)

import logging
from src.helpers.args import get_args, ModelArguments, DataTrainingArguments
from src.helpers.fn import (
    setup_logger,
    from_checkpoint,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from src.mtl.MTTokenizerPrompt import (
    convert_to_hotpotqa_features,
    convert_to_nq_features,
    convert_to_wikiLin_features,
    convert_to_xsum_features,
    convert_to_SeqClass_features_CG,
    convert_to_boolq_features_CG,
    convert_to_winogrande_features_CG,
    convert_to_mnli_features_CG,
    convert_to_anli_features_CG,
    convert_to_go_emotion_simple_features_CG,
    convert_to_piqa_features_CG,
    convert_to_socqa_features_CG,
    convert_to_SQUAD_features_CG,
    convert_to_tweetQA_features,
    convert_to_SeqClass_features_CG,
    convert_to_aeslc_features,
    convert_to_record_features,
    convert_to_qnli_features_CG,
)
from src.mtl.MTTrainer import MultitaskTrainer_OH
from src.mtl.MTSampler import Sampler

def prefinetune(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: Seq2SeqTrainingArguments,
    last_checkpoint : Union[None, str] = None
) -> Union[None, str]:
    """
    Arguments:
    Returns:
    """

    MAX_SOURCE_LENGTH = data_args.pf_max_source_length
    MAX_TARGET_LENGTH = data_args.pf_max_target_length

    import re
    task_families_to_exclude = []
    for tasks_rm in data_args.task_families_to_exclude:
        tasks_rm = tasks_rm.lower()
        for task_rm in tasks_rm.split():
            task_rm = re.sub('[^A-Za-z0-9 ]+', '', task_rm)
            task_families_to_exclude.append(task_rm)

    SIZE_PER_TASK = data_args.size_per_task  #14400
    SIZE_PER_TASK = SIZE_PER_TASK if (SIZE_PER_TASK > 0) else (data_args.num_stages * training_args.per_device_train_batch_size * ((data_args.num_task_families - len(task_families_to_exclude)) * data_args.num_tasks_per_family))
    SIZE_PER_FAMILY = SIZE_PER_TASK * data_args.num_tasks_per_family

    set_seed(training_args.seed)

    model_name = model_args.config_name if model_args.config_name else model_args.model_name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # -------------------------------------------------------------------------------------------------------
    # STEP 1: Dicts for dataset, model type and model config.
    # a datastructure to hold and pass all the datasets

    size_dict={
        "mlm": 50000,
        "cp": 50000,
        "sr": 50000,
        "sd": 50000,
        "imdb": 25000,
        "boolq": 9427,
        "squad": 87599,
        "winogrande": 9248,
        "mnli": 392702,
        "anli": 16946,
        "go_emotions": 43410,
        "hotpot_qa": 90447,
        "nq": 87925,
        "piqa": 16113,
        "socqa": 33410,
        "xsum": 204045,
        "tweet_qa": 10692,
        "ag_news": 120000,
        "aeslc": 14436,
        "record": 100730,
        "qnli": 104743,
        "wiki_lingua": 57945,
    }
    family_dict = {
        'go_emotions': 'cls',
        'imdb': 'cls',
        'ag_news': 'cls',
        'xsum': 'sum',
        'wiki_lingua': 'sum',
        'aeslc': 'sum',
        'boolq': 'rc',
        'squad': 'rc',
        'tweet_qa': 'rc',
        'hotpot_qa': 'cbqa',
        'nq': 'cbqa',
        'record': 'cbqa',
        'winogrande': 'cmns',
        'piqa': 'cmns',
        'socqa':'cmns',
        'mnli': 'nli',
        'anli': 'nli',
        'qnli': 'nli',
    }

    use_family_matrix = {
        'go_emotions': False if 'cls' in task_families_to_exclude else True,
        'imdb': False if 'cls' in task_families_to_exclude else True,
        'ag_news': False if 'cls' in task_families_to_exclude else True,
        'xsum': False if 'sum' in task_families_to_exclude else True,
        'wiki_lingua': False if 'sum' in task_families_to_exclude else True,
        'aeslc': False if 'sum' in task_families_to_exclude else True,
        'boolq': False if 'rc' in task_families_to_exclude else True,
        'squad': False if 'rc' in task_families_to_exclude else True,
        'tweet_qa': False if 'rc' in task_families_to_exclude else True,
        'hotpot_qa': False if 'cbqa' in task_families_to_exclude else True,
        'nq': False if 'cbqa' in task_families_to_exclude else True,
        'record': False if 'cbqa' in task_families_to_exclude else True,
        'winogrande': False if 'cmns' in task_families_to_exclude else True,
        'piqa':  False if 'cmns' in task_families_to_exclude else True,
        'socqa': False if 'cmns' in task_families_to_exclude else True,
        'mnli':  False if 'nli' in task_families_to_exclude else True,
        'anli': False if 'nli' in task_families_to_exclude else True,
        'qnli': False if 'nli' in task_families_to_exclude else True,
    }

    sampler = Sampler(family_dict, size_dict, limit=SIZE_PER_FAMILY, temperature=model_args.temperature)
    if model_args.family_scaling == "uniform":
        sampler.uniform()
        print("<< UNIFORM >>")
    elif model_args.family_scaling == "proportional":
        sampler.proportional()  # list of sizes per task
        print("<< PROPORTIONAL >>")
    elif model_args.family_scaling == "temperature":
        sampler.temperature()
        print("<< TEMPERATURE >>")

    dataset_dict = {  # uniform
        "go_emotions": sampler.load("go_emotions", 'go_emotions', addition='simplified') if use_family_matrix['go_emotions'] else None,
        "imdb": sampler.load("imdb", 'imdb') if use_family_matrix['imdb'] else None,
        "ag_news": sampler.load('ag_news','ag_news') if use_family_matrix['ag_news'] else None,
        "xsum": sampler.load("xsum", 'xsum') if use_family_matrix['xsum'] else None,
        "wiki_lingua": sampler.load('wiki_lingua', 'wiki_lingua', addition='english') if use_family_matrix['wiki_lingua'] else None,
        "aeslc": sampler.load('aeslc', 'aeslc') if use_family_matrix['aeslc'] else None,
        "boolq": sampler.load("boolq", 'boolq') if use_family_matrix['boolq'] else None,
        "squad": sampler.load("squad", 'squad') if use_family_matrix['squad'] else None,
        "tweet_qa": sampler.load('tweet_qa', 'tweet_qa', revision="master") if use_family_matrix['tweet_qa'] else None,
        "hotpot_qa": sampler.load("hotpot_qa", 'hotpot_qa', addition='fullwiki') if use_family_matrix['hotpot_qa'] else None,
        "nq": sampler.load("nq", 'nq_open') if use_family_matrix['nq'] else None,  # load_dataset('natural_questions'),
        "record": sampler.load('record', 'super_glue', addition='record') if use_family_matrix['record'] else None,
        "winogrande": sampler.load("winogrande", name='winogrande', addition='winogrande_debiased') if use_family_matrix['winogrande'] else None,
        "piqa": sampler.load("piqa", 'piqa') if use_family_matrix['piqa'] else None,
        "socqa": sampler.load("socqa", 'social_i_qa') if use_family_matrix['socqa'] else None,
        "mnli": sampler.load("mnli", 'glue', addition='mnli') if use_family_matrix['mnli'] else None,
        "anli": sampler.load("anli", 'anli', train='train_r1') if use_family_matrix['anli'] else None,
        "qnli": sampler.load('qnli','glue', addition='qnli') if use_family_matrix['qnli'] else None,
    }

    for ds, l in dataset_dict.items():
        print(f"{ds} : {l}")

    dataset_dict = {k:v for k, v in dataset_dict.items() if v != None}

    print(dataset_dict.keys())

    # Step 2: create MT Models
    # start the shared bare model creation.
    heterogeneous_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        config=transformers.AutoConfig.from_pretrained(model_name),
        )

    # Step 3: Tokenize datasets
    #
    convert_func_dict_cg = {
        "imdb": convert_to_SeqClass_features_CG(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "boolq": convert_to_boolq_features_CG(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "squad": convert_to_SQUAD_features_CG(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "winogrande": convert_to_winogrande_features_CG(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "mnli": convert_to_mnli_features_CG(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "anli": convert_to_anli_features_CG(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "go_emotions": convert_to_go_emotion_simple_features_CG(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "hotpot_qa": convert_to_hotpotqa_features(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "nq": convert_to_nq_features(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "piqa": convert_to_piqa_features_CG(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "socqa": convert_to_socqa_features_CG(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "xsum": convert_to_xsum_features(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "tweet_qa": convert_to_tweetQA_features(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "ag_news": convert_to_SeqClass_features_CG(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "aeslc": convert_to_aeslc_features(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "record": convert_to_record_features(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "qnli": convert_to_qnli_features_CG(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),
        "wiki_lingua": convert_to_wikiLin_features(tokenizer, length_source=MAX_SOURCE_LENGTH, length_target=MAX_TARGET_LENGTH),

    }

    print(f"\n\n{len(dataset_dict)}\n\n")

    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        print(f"TASK NAME: {task_name} with {convert_func_dict_cg[task_name]}")
        features_dict[task_name] = {}
        phase='train'
        phase_dataset=dataset
        features_dict[task_name][phase] = phase_dataset.map(
            convert_func_dict_cg[task_name],
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
        features_dict[task_name][phase].set_format(
            type="torch",
            columns=['input_ids', 'attention_mask', 'labels'],
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))

    # Step 4: Training
    # we use our customized multitask trainer to feed a sequence of different tasks to the models

    train_dataset = {
        task_name: dataset['train'] if 'train' in dataset.keys() else dataset['train_r1'] for task_name, dataset in features_dict.items()
    }

    print("PRE SAFE +++++++++++++++++++++++++")
    pre_train_save_dir = training_args.output_dir
    import os
    if not os.path.exists(pre_train_save_dir):
        os.makedirs(pre_train_save_dir)
    tokenizer.save_vocabulary(pre_train_save_dir)
    tokenizer.save_pretrained(pre_train_save_dir)

    trainer =  MultitaskTrainer_OH(
        model=heterogeneous_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=transformers.data.data_collator.torch_default_data_collator,
    )

    train_result = trainer.train()

    # Step 5: Checkpoint for Finetuning
    #
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

    new_checkpoint = from_checkpoint(training_args=training_args, logger=logger)
    return new_checkpoint



def main_repl(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: Seq2SeqTrainingArguments,
    logger : logging.Logger,
) -> None:
    """
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

    # Load the last checkpoint if possible
    last_checkpoint = from_checkpoint(training_args=training_args, logger=logger)

    if (data_args.prefinetune):
        last_checkpoint = prefinetune(model_args, data_args, training_args, last_checkpoint)

if __name__ == "__main__":
    check_min_version("4.9.2")
    require_version("datasets>=1.8.0", "To fix: pipenv install.")
    logger = logging.getLogger(__name__)
    model_args, data_args, training_args = get_args()
    main_repl(model_args, data_args, training_args, logger)
