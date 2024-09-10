import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer, PaddingStrategy


def prepare_dataset(
    dataset,
    tokenizer,
    train_on_response=False,
    separator=None,
):
    """
    Tokenize the examples and prepare the dataset for training.

    If separator is None, tokenize the full prompt and set labels based on train_on_response.
    If separator is provided, use chat template and set labels based on the separator position.
    """

    def tokenize_function(example):
        if separator is None:
            example["messages"][-1]["content"] = (
                example["messages"][-1]["content"] + tokenizer.eos_token
            )
        if separator is None:
            full_prompt = "".join(msg["content"] for msg in example["messages"])
            tokenized = tokenizer(full_prompt, truncation=True)

            if train_on_response:
                user_token_count = len(
                    tokenizer(example["messages"][0]["content"])["input_ids"]
                )
                labels = [-100] * user_token_count + tokenized["input_ids"][
                    user_token_count:
                ]
            else:
                labels = tokenized["input_ids"].copy()
        else:
            formatted_prompt = tokenizer.apply_chat_template(
                example["messages"], tokenize=False
            ).replace(tokenizer.bos_token, "")
            tokenized = tokenizer(formatted_prompt, truncation=True)

            separator_tokens = tokenizer.encode(separator, add_special_tokens=False)
            separator_index = find_subsequence(tokenized["input_ids"], separator_tokens)

            if separator_index != -1:
                labels = [-100] * (separator_index + len(separator_tokens)) + tokenized[
                    "input_ids"
                ][separator_index + len(separator_tokens) :]
            else:
                labels = [-100] * len(tokenized["input_ids"])

        labels = [label for label in labels]

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    def find_subsequence(seq, subseq):
        n, m = len(seq), len(subseq)
        for i in range(n - m + 1):
            if seq[i : i + m] == subseq:
                return i
        return -1

    return dataset.map(tokenize_function, num_proc=os.cpu_count()).select_columns(
        ["input_ids", "attention_mask", "labels"]
    )


@dataclass
class SFTDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if "labels" in batch:
            # Replace padding token id in the labels by -100
            batch["labels"] = [
                [
                    (label if label != self.tokenizer.pad_token_id else -100)
                    for label in labels
                ]
                for labels in batch["labels"]
            ]
            batch["labels"] = torch.tensor(batch["labels"])

        return batch
