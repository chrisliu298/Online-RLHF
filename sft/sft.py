import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from chat_templates import chat_templates
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTTrainer


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, \
    what their capacity and features are, and what size model you want to train.
    """

    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.0)
    model_name: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_name: Optional[str] = field(
        default="RLHFlow/SFT-OpenHermes-2.5-Standard",
        metadata={
            "help": "",
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[float] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        default="paged_adamw_32bit",
        # default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_training_samples: Optional[int] = field(
        default=-1, metadata={"help": "the maximum sample size"}
    )
    max_length: Optional[int] = field(default=4096)
    output_dir: Optional[str] = field(default="./models/sft_model_llama3")
    use_liger_kernel: Optional[bool] = field(
        default=False,
        metadata={"help": "Use liger kernel for faster training."},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    save_strategy="epoch",
    save_steps=1000000000,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    remove_unused_columns=True,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.03,
    report_to="wandb",
    use_liger_kernel=script_args.use_liger_kernel,
)


dataset = load_from_disk(script_args.dataset_name)

if script_args.max_training_samples > 0:
    dataset = dataset.select(range(script_args.max_training_samples))

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("We set the pad token as the eos token by default....")
tokenizer.model_max_length = script_args.max_length
tokenizer.chat_template = chat_templates[script_args.model_name.split("/")[-1]]


def formatting_prompts_func(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False
        ).replace(tokenizer.bos_token, "")
    }


ds = dataset.map(formatting_prompts_func, num_proc=os.cpu_count())
# collator = DataCollatorForCompletionOnlyLM(
#     response_template="</score><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
#     tokenizer=tokenizer,
# )

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=script_args.max_length,
    packing=True,
    # data_collator=collator,
    # dataset_num_proc=os.cpu_count(),
)

trainer.train()
print("Saving last checkpoint of the model")

trainer.save_model(script_args.output_dir)
tokenizer.save_pretrained(script_args.output_dir)
