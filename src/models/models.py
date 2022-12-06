from typing import Tuple
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from src.helpers.args import ModelArguments


def get_model_fn_wrapper(model_args: ModelArguments, tokenizer):
    def get_model_fn() -> AutoModelForSeq2SeqLM:
        """Retrieve the model.

        Args:
            model_args (ModelArguments): [Arguments for the model]

        Returns:
            AutoModelForSeq2SeqLM: The model
        """

        # Load pretrained model and tokenizer
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model.resize_token_embeddings(len(tokenizer))

        if model.config.decoder_start_token_id is None:  # pragma: no cover
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
        return model

    return get_model_fn


def get_tokenizer(model_args: ModelArguments) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
