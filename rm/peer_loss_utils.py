from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import AutoTokenizer, Trainer
from transformers.utils import PaddingStrategy


def build_dataset_local(tokenizer, train_path, tokenize=True):
    def tokenize_func(sample):
        sample["positive"] = tokenizer.apply_chat_template(
            sample["chosen"], tokenize=False, add_generation_prompt=False
        )
        sample["negative"] = tokenizer.apply_chat_template(
            sample["rejected"], tokenize=False, add_generation_prompt=False
        )
        sample["alt"] = tokenizer.apply_chat_template(
            sample["alt"], tokenize=False, add_generation_prompt=False
        )
        if tokenizer.bos_token is not None:
            sample["positive"] = sample["positive"].replace(tokenizer.bos_token, "")
            sample["negative"] = sample["negative"].replace(tokenizer.bos_token, "")
            sample["alt"] = sample["alt"].replace(
                tokenizer.bos_token, ""
            )
        tokenized_pos = tokenizer(sample["positive"], truncation=True)
        tokenized_neg = tokenizer(sample["negative"], truncation=True)
        tokenized_alt_neg = tokenizer(sample["alt"], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
        sample["input_ids_l"] = tokenized_alt_neg["input_ids"]
        sample["attention_mask_l"] = tokenized_alt_neg["attention_mask"]
        return sample

    dataset = load_from_disk(train_path).shuffle(seed=42)
    if not tokenize:
        return dataset
    dataset = dataset.map(tokenize_func, num_proc=8)
    return dataset


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_l"],
                    "attention_mask": feature["attention_mask_l"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }
        return batch


# Define the trainer
def compute_metrics(eval_pred):
    result = {}
    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]
    # We assume that the first sample is preferred by default in groundtruth
    result["accuracy"] = np.sum(pos_predictions_scores > neg_predictions_scores) / len(
        pos_predictions_scores
    )
    return result


class RewardTrainer(Trainer):
    def __init__(self, peer_loss_lambda=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peer_loss_lambda = peer_loss_lambda

    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 3)
        kidx = jidx + 1
        lidx = jidx + 2
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        rewards_l = rewards[lidx]
        loss = (
            -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
            + self.peer_loss_lambda
            * nn.functional.logsigmoid(rewards_j - rewards_l).mean()
        )
        if return_outputs:
            return loss, {
                "rewards_j": rewards_j,
                "rewards_k": rewards_k,
                "rewards_l": rewards_l,
            }

        # Log the rewards
        # self.log("rewards/chosen", rewards_j.mean().item())
        # self.log("rewards/rejected", rewards_k.mean().item())
        # self.log("rewards/alt", rewards_l.mean().item())
        self.log({"rewards/chosen": rewards_j.mean().item()})
        self.log({"rewards/rejected": rewards_k.mean().item()})
        self.log({"rewards/alt": rewards_l.mean().item()})
        return loss
