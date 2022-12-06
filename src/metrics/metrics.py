from typing import List, Dict, Tuple, Callable, Union

import torch
import numpy as np

from transformers import PreTrainedTokenizer

from datasets import load_metric

from src.helpers.fn import postprocess_text_rouge_meteor, postprocess_text_bleu
from src.helpers.args import DataTrainingArguments

bertscore = load_metric("bertscore")
rouge = load_metric("rouge")
meteor = load_metric("meteor")
bleu = load_metric("bleu")

metric_function_map = {
    "bertscore": bertscore,
    "rouge": rouge,
    "meteor": meteor,
    "bleu": bleu,
}


def create_compute_metric_fn(
    tokenizer: PreTrainedTokenizer,
    data_args: DataTrainingArguments,
    evaluation_strategies: List[str],
) -> Callable[[Tuple[torch.Tensor, torch.Tensor]], Dict[str, Dict[str, float]]]:
    """Creates the metric function to evaluate during evaluation and prediction.

    Args:
        tokenizer (PreTrainedTokenizer): [The tokenizer.]
        data_args (DataTrainingArguments): [DataTrainingArguments.]
        evaluation_strategies (List[str]): [Which metrics to use. See DataTrainingArguments.]

    Returns:
        Callable[[Tuple[torch.Tensor, torch.Tensor]], Dict[str, Dict[str, float]]]: [Returns a
        callback function doing the metric calculation and returning the results]
    """

    def compute_metrics_fn(
        eval_preds: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """A function calculating the the metrics given predictions and labels. Uses the eval
        strategies defined in the outer funciton create_compute_metric_fn.

        Args:
            eval_preds (Tuple[torch.Tensor, torch.Tensor]): [A tuple of predictions and labels.]

        Returns:
            Dict[str, Dict[str, float]]: [A dictionary containing keys and metric values.]
        """
        results: Dict[str, Dict[str, float]] = dict()

        preds, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        for evaluation_strategy in evaluation_strategies:
            metric = metric_function_map[evaluation_strategy]

            kwargs: Dict[str, Union[str, bool, List[str]]] = dict()

            metric_preds = metric_labels = None

            if evaluation_strategy == "bertscore":
                kwargs.update({"lang": "en"})
                metric_preds, metric_labels = decoded_preds, decoded_labels
            if evaluation_strategy[:5] == "rouge":
                kwargs.update(
                    {
                        "rouge_types": ["rouge1", "rouge2", "rougeL"],
                        "use_stemmer": True,
                    }
                )
            if (
                evaluation_strategy[:5] == "rouge"
                or evaluation_strategy == "meteor"
                or evaluation_strategy == "bertscore"
            ):
                metric_preds, metric_labels = postprocess_text_rouge_meteor(
                    decoded_preds, decoded_labels
                )
            if evaluation_strategy == "bleu":
                metric_preds, metric_labels = postprocess_text_bleu(decoded_preds, decoded_labels)

            new_result = metric.compute(
                predictions=metric_preds, references=metric_labels, **kwargs
            )

            if new_result:
                if evaluation_strategy == "rouge":
                    # Rouge provides high, low, and mid measures.
                    # We take the standard mid measure here
                    new_result = {key: value.mid.fmeasure for key, value in new_result.items()}
                elif evaluation_strategy == "bertscore":
                    # bertscore provides a list of f1 scores for each summary.
                    # We take the mean f1 score.
                    new_result = {"bertscore": np.mean(new_result["f1"])}
                elif evaluation_strategy == "bleu":
                    # Bleu just provides one score and we take that.
                    new_result = {"bleu": new_result["bleu"]}
                else:
                    # Otherwise we take the official metric
                    new_result = dict(new_result.items())
                results.update(new_result)

        return results

    return compute_metrics_fn
