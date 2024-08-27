from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, Trainer
from transformers.utils import PaddingStrategy


def build_dataset(tokenizer, train_path):
    def tokenize(sample):
        sample["positive"] = tokenizer.apply_chat_template(
            sample["chosen"], tokenize=False, add_generation_prompt=False
        )
        sample["negative"] = tokenizer.apply_chat_template(
            sample["rejected"], tokenize=False, add_generation_prompt=False
        )
        if tokenizer.bos_token is not None:
            sample["positive"] = sample["positive"].replace(tokenizer.bos_token, "")
            sample["negative"] = sample["negative"].replace(tokenizer.bos_token, "")
        tokenized_pos = tokenizer(sample["positive"], truncation=True)
        tokenized_neg = tokenizer(sample["negative"], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
        return sample

    dataset = load_dataset(train_path, split="train").shuffle(seed=42)
    dataset = dataset.map(tokenize, num_proc=8)
    return dataset


def build_dataset_local(tokenizer, train_path, tokenize=True):
    def tokenize_func(sample):
        sample["positive"] = tokenizer.apply_chat_template(
            sample["chosen"], tokenize=False, add_generation_prompt=False
        )
        sample["negative"] = tokenizer.apply_chat_template(
            sample["rejected"], tokenize=False, add_generation_prompt=False
        )
        if tokenizer.bos_token is not None:
            sample["positive"] = sample["positive"].replace(tokenizer.bos_token, "")
            sample["negative"] = sample["negative"].replace(tokenizer.bos_token, "")
        tokenized_pos = tokenizer(sample["positive"], truncation=True)
        tokenized_neg = tokenizer(sample["negative"], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
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


def t_log(x, t):
    return (x ** (1 - t) - 1) / (1 - t)


def t_log_sigmoid(x, t):
    return t_log(torch.sigmoid(x), t)


class RewardTrainer(Trainer):
    def __init__(
        self, loss_type="bt", log_t=1.0, gamma=0.0, margin=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert loss_type in {
            "bt",
            "t_log",
            "focal",
            "hinge",
            "margin_mse",
            "backward_ce",
            "forward_ce",
            "ce",
            "sim",
        }, f"Invalid loss type: {loss_type}"
        self.loss_type = loss_type
        self.log_t = log_t
        self.gamma = gamma
        self.margin = margin

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_type not in {"sim", "sim_per_layer"}:
            rewards = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )[0]
        elif self.loss_type == "sim":
            outputs = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            rewards = outputs[0]
            last_hidden_state = outputs.hidden_states[-1][:, -1, :].view(
                rewards.size(0), -1
            )
        elif self.loss_type == "sim_per_layer":
            outputs = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            rewards = outputs[0]
            # last_hidden_state = outputs.hidden_states[-1][:, -1, :].view(
            #     rewards.size(0), -1
            # )
            hidden_states = outputs.hidden_states[1:]
            last_hidden_states = [
                h[:, -1, :].view(rewards.size(0), -1) for h in hidden_states
            ]

        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]

        if self.loss_type == "bt":
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        elif self.loss_type == "t_log":
            loss = -t_log_sigmoid(rewards_j - rewards_k, self.log_t).mean()
        elif self.loss_type == "focal":
            loss = (
                -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
                * (1 - torch.sigmoid(rewards_j - rewards_k)).mean() ** self.gamma
            )
        elif self.loss_type == "hinge":
            loss = torch.relu(self.margin - (rewards_j - rewards_k)).mean()
        elif self.loss_type == "margin_mse":
            loss = nn.functional.mse_loss(rewards_j, rewards_k + self.margin)
        elif self.loss_type == "backward_ce":
            logits = rewards_j - rewards_k
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, torch.ones_like(logits)
            )
        elif self.loss_type == "forward_ce":
            logits = rewards_j - rewards_k
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, torch.zeros_like(logits)
            )
        elif self.loss_type == "ce":
            labels_j = torch.ones_like(rewards_j)
            labels_k = torch.zeros_like(rewards_k)
            logits = torch.cat([rewards_j, rewards_k], dim=0)
            labels = torch.cat([labels_j, labels_k], dim=0)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        elif self.loss_type == "sim":
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
            chosen_hidden_states = last_hidden_state[jidx]
            rejected_hidden_states = last_hidden_state[kidx]
            sim = torch.cosine_similarity(chosen_hidden_states, rejected_hidden_states)
            self.log({"sim": sim.mean().item()})
            loss += sim.mean() * self.gamma
        elif self.loss_type == "sim_per_layer":
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
            chosen_hidden_states = []
            rejected_hidden_states = []
            sim = 0.0
            for h in last_hidden_states:
                sim += torch.cosine_similarity(h[jidx], h[kidx])
            self.log({"sim": sim.mean().item()})
            loss += sim.mean() * self.gamma

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}

        self.log({"rewards/chosen": rewards_j.mean().item()})
        self.log({"rewards/rejected": rewards_k.mean().item()})

        return loss
