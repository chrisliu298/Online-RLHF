import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Trainer
from transformers.trainer_utils import seed_worker
from transformers.utils import PaddingStrategy, is_datasets_available

special_tokens = {
    "gemma-2-27b-it": "<unused0>",
    "Meta-Llama-3.1-8B-Instruct": "<|reserved_special_token_0|>",
}


def check_valid_checkpoint(base_path, world_size):
    # Define common static files and rng state files
    static_files_common = [
        "config.json",
        "latest",
        "model.safetensors.index.json",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        "zero_to_fp32.py",
    ]

    rng_state_files = [f"rng_state_{i}.pth" for i in range(world_size)]

    # Determine model-specific files
    if "Meta-Llama-3.1-8B-Instruct" in base_path:
        model_files = [
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ]

        optim_state_files = [
            f"bf16_zero_pp_rank_{i}_mp_rank_00_optim_states.pt"
            for i in range(world_size)
        ]
        model_state_files = ["mp_rank_00_model_states.pt"]

    elif "gemma-2-27b-it" in base_path:
        model_files = [
            "model-00001-of-00012.safetensors",
            "model-00002-of-00012.safetensors",
            "model-00003-of-00012.safetensors",
            "model-00004-of-00012.safetensors",
            "model-00005-of-00012.safetensors",
            "model-00006-of-00012.safetensors",
            "model-00007-of-00012.safetensors",
            "model-00008-of-00012.safetensors",
            "model-00009-of-00012.safetensors",
            "model-00010-of-00012.safetensors",
            "model-00011-of-00012.safetensors",
            "model-00012-of-00012.safetensors",
        ]

        optim_state_files = [
            f"bf16_zero_pp_rank_{i}_mp_rank_00_optim_states.pt"
            for i in range(world_size)
        ]
        model_state_files = [
            f"zero_pp_rank_{i}_mp_rank_00_model_states.pt" for i in range(world_size)
        ]

    else:
        print("Invalid model_name provided.")
        return False

    # The pattern for the global_step folder
    global_step_folder_pattern = "global_step"

    # Find the global_step folder
    global_step_folder = None
    for item in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, item)) and item.startswith(
            global_step_folder_pattern
        ):
            global_step_folder = item
            break

    if global_step_folder is None:
        print(f"No global_step folder found for {base_path}.")
        return False

    # Complete paths for the optim state and model state files
    optim_state_files = [
        os.path.join(global_step_folder, file) for file in optim_state_files
    ]
    model_state_files = [
        os.path.join(global_step_folder, file) for file in model_state_files
    ]

    # Combine all file lists
    all_files = (
        static_files_common
        + model_files
        + rng_state_files
        + optim_state_files
        + model_state_files
    )

    # Check if all files exist
    missing_files = [
        file for file in all_files if not os.path.exists(os.path.join(base_path, file))
    ]

    if missing_files:
        print("The following files are missing:")
        for file in missing_files:
            print(file)
        return False

    print("All files are present.")
    return True


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


def build_dataset_local(
    tokenizer, train_path, tokenize=True, special_token="", add_system_prompt=False
):
    def tokenize_func(sample):
        if add_system_prompt:
            system_prompt = {
                "role": "system",
                "content": "You are a good reward model. Please accurately score the provided response based on helpfulness, correctness, coherence, complexity, and verbosity.",
            }
            sample["positive"] = (
                tokenizer.apply_chat_template(
                    [system_prompt] + sample["chosen"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                + special_token
            )
            sample["negative"] = (
                tokenizer.apply_chat_template(
                    [system_prompt] + sample["rejected"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                + special_token
            )
        else:
            sample["positive"] = (
                tokenizer.apply_chat_template(
                    sample["chosen"], tokenize=False, add_generation_prompt=False
                )
                + special_token
            )
            sample["negative"] = (
                tokenizer.apply_chat_template(
                    sample["rejected"], tokenize=False, add_generation_prompt=False
                )
                + special_token
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
    # Add average reward scores
    result["rewards/chosen"] = np.mean(pos_predictions_scores)
    result["rewards/rejected"] = np.mean(neg_predictions_scores)
    return result


def t_log(x, t):
    return (x ** (1 - t) - 1) / (1 - t)


def t_log_sigmoid(x, t):
    return t_log(torch.sigmoid(x), t)


class RewardTrainer(Trainer):
    def __init__(
        self,
        loss_type="bt",
        temp=1.0,
        log_t=1.0,
        gamma=0.0,
        margin=1.0,
        lambd=0.0,
        log_reward=False,
        reward_range=None,
        *args,
        **kwargs,
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
            "sim_per_layer",
            "bt_per_layer",
            "focal_penalty",
        }, f"Invalid loss type: {loss_type}"
        self.loss_type = loss_type
        self.temp = temp
        self.log_t = log_t
        self.gamma = gamma
        self.lambd = lambd
        self.margin = margin
        self.log_reward = log_reward
        self.reward_range = reward_range

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_type not in {"sim", "sim_per_layer", "bt_per_layer"}:
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
            hidden_states = outputs.hidden_states[1:]
            last_hidden_states = [
                h[:, -1, :].view(rewards.size(0), -1) for h in hidden_states
            ]
        elif self.loss_type == "bt_per_layer":
            outputs = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            rewards = outputs[0]
            hidden_states = outputs.hidden_states[:-1]
            assert len(hidden_states) == len(model.model.layers)
            intermediate_rewards = [model.score(h) for h in hidden_states]

        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]

        if self.reward_range is not None:
            penalty_j = torch.where(
                rewards_j > self.reward_range[1],
                (rewards_j - self.reward_range[1]) ** 2,
                torch.where(
                    rewards_j < self.reward_range[0],
                    (rewards_j - self.reward_range[0]) ** 2,
                    0,
                ),
            ).mean()
            penalty_k = torch.where(
                rewards_k > self.reward_range[1],
                (rewards_k - self.reward_range[1]) ** 2,
                torch.where(
                    rewards_k < self.reward_range[0],
                    (rewards_k - self.reward_range[0]) ** 2,
                    0,
                ),
            ).mean()

        if self.loss_type == "bt":
            rewards_diff = rewards_j - rewards_k
            if self.temp != 1.0:
                loss = -nn.functional.logsigmoid(rewards_diff / self.temp).mean()
            else:
                loss = -nn.functional.logsigmoid(rewards_diff).mean()
        elif self.loss_type == "t_log":
            loss = -t_log_sigmoid(rewards_j - rewards_k, self.log_t).mean()
        elif self.loss_type == "focal":
            loss = (
                -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
                * (1 - torch.sigmoid(rewards_j - rewards_k)).mean() ** self.gamma
            )
        elif self.loss_type == "focal_penalty":
            margin_prob = torch.sigmoid(rewards_j - rewards_k)
            zeros = torch.zeros_like(margin_prob)
            # Stack in the 0th dimension
            margin_prob_zeros = torch.cat([margin_prob - 0.5, zeros], dim=0)
            ranking_loss = (
                -((1 - 2 * torch.max(margin_prob_zeros, dim=0).values) ** self.gamma)
                * torch.log(margin_prob)
            ).mean()
            # penalty_j = -(torch.log(rewards_j + 5) + torch.log(5 - rewards_j)).mean()
            # penalty_k = -(torch.log(rewards_k + 5) + torch.log(5 - rewards_k)).mean()
            loss = ranking_loss  # + self.lambd * penalty_j + self.lambd * penalty_k
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
                sim += torch.cosine_similarity(h[jidx], h[kidx]) / len(
                    last_hidden_states
                )
            self.log({"sim": sim.mean().item()})
            loss += sim.mean() * self.gamma
        elif self.loss_type == "bt_per_layer":
            last_layer_loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
            intermediate_losses = []
            for i, r in enumerate(intermediate_rewards):
                layer_loss = -nn.functional.logsigmoid(r[jidx] - r[kidx]).mean()
                intermediate_losses.append(layer_loss)
            intermediate_loss = torch.stack(intermediate_losses).sum()
            loss = (last_layer_loss + intermediate_loss) / (
                1 + len(intermediate_losses)
            )

        if self.reward_range is not None:
            loss += self.lambd * penalty_j + self.lambd * penalty_k

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}

        if self.log_reward:
            self.log({"rewards/chosen": rewards_j.mean().item()})
            self.log({"rewards/rejected": rewards_k.mean().item()})

        return loss
