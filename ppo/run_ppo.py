import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl.trainer.ppov2_trainer import PPOv2Config, PPOv2Trainer


@dataclass
class ScriptArguments:
    """
    The arguments for the PPO training script.
    """

    # training parameters
    dataset_name_or_path: Optional[str] = field(
        default="prompt-collection-v0.1",
        metadata={"help": "the location of the dataset name or path"},
    )
    model_name_or_path: Optional[str] = field(
        default="gemma-2b-it",
        metadata={"help": "the location of the model name or path"},
    )
    reward_model_name_or_path: Optional[str] = field(
        default="gemma-2b-it-rm",
        metadata={"help": "the location of the reward model name or path"},
    )
    learning_rate: Optional[float] = field(
        default=1e-6, metadata={"help": "optimizer learning rate"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "the lr scheduler type"}
    )
    warmup_ratio: Optional[float] = field(
        default=0.1, metadata={"help": "the percentage of warmup steps"}
    )
    weight_decay: Optional[float] = field(
        default=0.0, metadata={"help": "the weight decay"}
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit", metadata={"help": "the optimizer type"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "train batch size per device"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "eval batch size per device"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=64, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    eos_padding: Optional[bool] = field(
        default=True, metadata={"help": "whether to pad with eos token"}
    )
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "max number of training epochs"}
    )
    num_ppo_epochs: Optional[int] = field(
        default=1, metadata={"help": "max number of PPO epochs"}
    )
    logging_steps: Optional[int] = field(
        default=10, metadata={"help": "the logging frequency"}
    )
    save_steps: Optional[int] = field(
        default=999999, metadata={"help": "the saving frequency"}
    )
    eval_steps: Optional[int] = field(
        default=100, metadata={"help": "the evaluation frequency"}
    )
    # instrumentation
    num_training_samples: Optional[int] = field(
        default=-1, metadata={"help": "the number of training sample size"}
    )
    num_eval_samples: Optional[int] = field(
        default=128, metadata={"help": "the number of eval sample size"}
    )
    bf16: Optional[bool] = field(
        default=True, metadata={"help": "whether to use bfloat16"}
    )
    # output directory
    output_dir: Optional[str] = field(
        default="ppo_models", metadata={"help": "the output directory"}
    )
    num_sample_generations: Optional[int] = field(
        default=1, metadata={"help": "the number of sample generations"}
    )
    local_rollout_forward_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the number of forward rollouts"}
    )
    prompt_response_length: Optional[int] = field(
        default=4096, metadata={"help": "the maximum prompt/response length"}
    )

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = f"./ppo_models/{self.model_name}-ppo"


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# 1. load a pretrained model
policy_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
policy_model.config.use_cache = False

ref_name = script_args.model_name_or_path
reference_model = AutoModelForCausalLM.from_pretrained(
    ref_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    script_args.reward_model_name_or_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
value_model = AutoModelForSequenceClassification.from_pretrained(
    script_args.reward_model_name_or_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

if tokenizer.pad_token is None:
    if script_args.eos_padding:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set padding token to eos token: {tokenizer.pad_token}")
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        policy_model.config.vocab_size += 1
        reference_model.config.vocab_size += 1
        policy_model.config.pad_token_id = tokenizer.pad_token_id
        reference_model.config.pad_token_id = tokenizer.pad_token_id
        policy_model.resize_token_embeddings(len(tokenizer))
        reference_model.resize_token_embeddings(len(tokenizer))
        print(f"Added padding token: {tokenizer.pad_token}")

tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.prompt_response_length


def tokenize(sample):
    formatted = tokenizer.apply_chat_template(
        sample["context_messages"], tokenize=False, add_generation_prompt=True
    ).replace(tokenizer.bos_token, "")
    tokenized = tokenizer(formatted, truncation=True)
    return tokenized


# 2. Load the Stack-exchange paired dataset
rng = np.random.default_rng(seed=42)
full_dataset = load_dataset(script_args.dataset_name_or_path, split="train")
full_indices = rng.choice(len(full_dataset), len(full_dataset), replace=False)
assert (
    len(full_dataset) - script_args.num_training_samples > script_args.num_eval_samples
), f"Number of training samples {len(full_dataset) - script_args.num_training_samples} is less than number of evaluation samples {script_args.num_eval_samples}"
train_indices = rng.choice(
    len(full_dataset), script_args.num_training_samples, replace=False
)
train_dataset = full_dataset.select(train_indices)
rest_indices = np.setdiff1d(full_indices, train_indices)
eval_indices = rng.choice(rest_indices, script_args.num_eval_samples, replace=False)
eval_dataset = full_dataset.select(eval_indices)
print(f"Using {len(train_dataset)} training samples")
print(f"Using {len(eval_dataset)} evaluation samples")
train_dataset = train_dataset.map(
    tokenize,
    num_proc=os.cpu_count(),
    remove_columns=["dataset", "context", "context_messages", "id"],
)
eval_dataset = eval_dataset.map(
    tokenize,
    num_proc=os.cpu_count(),
    remove_columns=["dataset", "context", "context_messages", "id"],
)

# 4. initialize training arguments:
ppo_config = PPOv2Config(
    # PPOv2 arguments
    num_mini_batches=1,
    total_episodes=None,
    local_rollout_forward_batch_size=script_args.local_rollout_forward_batch_size,
    num_sample_generations=script_args.num_sample_generations,
    base_model=script_args.model_name_or_path,
    response_length=script_args.prompt_response_length,
    stop_token=None,
    stop_token_id=None,
    temperature=0.7,
    penalty_reward_value=-10.0,
    non_eos_penalty=True,
    reward_model_path=script_args.reward_model_name_or_path,
    sft_model_path=script_args.model_name_or_path,
    num_ppo_epochs=script_args.num_ppo_epochs,
    vf_coef=0.1,
    cliprange=0.2,
    cliprange_value=0.2,
    gamma=1.0,
    lam=0.95,
    whiten_rewards=False,
    kl_coef=0.05,  # 0.0325 for large reward models
    # Training arguments
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    learning_rate=script_args.learning_rate,
    weight_decay=script_args.weight_decay,
    max_grad_norm=1.0,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-5,
    save_steps=script_args.save_steps,
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=script_args.logging_steps,
    eval_steps=script_args.eval_steps,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=script_args.warmup_ratio,
    report_to="wandb",
)

# 5. initialize the PPO trainer
ppo_trainer = PPOv2Trainer(
    config=ppo_config,
    tokenizer=tokenizer,
    policy=policy_model,
    ref_policy=reference_model,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=train_dataset,
    eval_dataset=None,
)
print("Training started")

# 6. train
ppo_trainer.train()
ppo_trainer.save_model(script_args.output_dir)

# 7. save policy model and value model
policy_model.save_pretrained(os.path.join(script_args.output_dir, "policy"))
tokenizer.save_pretrained(os.path.join(script_args.output_dir, "policy"))
value_model.save_pretrained(os.path.join(script_args.output_dir, "value"))
tokenizer.save_pretrained(os.path.join(script_args.output_dir, "value"))
