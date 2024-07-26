from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from utils import (
    RewardDataCollatorWithPadding,
    RewardTrainer,
    build_dataset,
    build_dataset_local,
    compute_metrics,
)


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=64)
    learning_rate: Optional[float] = field(default=1e-5)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="Meta-Llama-3-8B-Instruct",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_set_path: Optional[str] = field(
        default="preference_dataset_mixture2_and_safe_pku",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    eval_set_path: Optional[str] = field(
        default="preference_dataset_mixture2_and_safe_pku",
        metadata={"help": "The dir of the subset of the evaluation data to use"},
    )
    output_dir: Optional[str] = field(
        default="./bt_models", metadata={"help": "The dir for output model"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Enables gradient checkpointing."}
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to use."}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "The lr scheduler"}
    )
    warmup_ratio: Optional[float] = field(
        default=0.0, metadata={"help": "The warmup ratio"}
    )
    warmup_steps: Optional[int] = field(
        default=0, metadata={"help": "The warmup steps"}
    )
    max_length: Optional[int] = field(default=4096)
    save_every_steps: Optional[int] = field(
        default=999999, metadata={"help": "Save the model every x steps"}
    )
    load_data_from_local: Optional[bool] = field(
        default=False, metadata={"help": "Load the data from local disk"}
    )
    eval_strategy: Optional[str] = field(
        default="steps", metadata={"help": "The evaluation strategy"}
    )
    eval_steps: Optional[int] = field(
        default=500, metadata={"help": "The evaluation steps"}
    )
    run_name: Optional[str] = field(
        default="reward_modeling", metadata={"help": "The name of the run"}
    )
    logging_steps: Optional[int] = field(
        default=1, metadata={"help": "The logging steps"}
    )

    def __post_init__(self):
        if self.output_dir == "./bt_models":
            if "/" in self.train_set_path:
                train_set_path = self.train_set_path.split("/")[0]
            else:
                train_set_path = self.train_set_path
            self.output_dir = f"./bt_models/{self.model_name}_{train_set_path}_{self.num_train_epochs}_{self.learning_rate}_{self.lr_scheduler_type}"


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the value-head model and tokenizer.
tokenizer_name = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

# has_no_pad_token = tokenizer.pad_token is None
# if has_no_pad_token:
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length

# Get the dataset
train_path = script_args.train_set_path
eval_path = script_args.eval_set_path
output_name = script_args.output_dir
if script_args.load_data_from_local:
    train_dataset = build_dataset_local(tokenizer, train_path)
    eval_dataset = build_dataset_local(tokenizer, eval_path)
else:
    train_dataset = build_dataset(tokenizer, train_path)
    eval_dataset = build_dataset(tokenizer, eval_path)
print("Training set:", len(train_dataset))
print("Evaluation set:", len(eval_dataset))

# Define the trainer
training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    save_strategy="steps",
    save_steps=script_args.save_every_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=script_args.logging_steps,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=script_args.warmup_ratio,
    warmup_steps=script_args.warmup_steps,
    report_to="wandb",
    eval_strategy=script_args.eval_strategy,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    eval_steps=script_args.eval_steps,
    run_name=script_args.run_name,
)

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = not script_args.gradient_checkpointing
# if has_no_pad_token:
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))

trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=script_args.max_length
    ),
)
trainer.train()
trainer.save_model(output_name)
tokenizer.save_pretrained(output_name)
