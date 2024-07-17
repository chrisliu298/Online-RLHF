import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_from_disk
from dpo import PreferenceTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import DPOConfig


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters, i.e., the KL penalty in the paper
    beta: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="/mnt/data2/yuhaoliu/sft_models/Qwen2-7B_SFT-OpenHermes-2.5-Standard",
        metadata={"help": "the location of the model name or path"},
    )
    ref_model: Optional[str] = field(
        default="",
        metadata={"help": "the location of the SFT model name or path"},
    )
    train_dir: Optional[str] = field(
        default="/mnt/data2/yuhaoliu/hf_datasets/UltraFeedback-preference-standard_processed",
        metadata={"help": "the location of the dataset name or path"},
    )
    learning_rate: Optional[float] = field(
        default=5e-7, metadata={"help": "optimizer learning rate"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "the lr scheduler type"}
    )
    weight_decay: Optional[float] = field(
        default=0.01, metadata={"help": "the weight decay"}
    )
    optimizer_type: Optional[str] = field(
        default="paged_adamw_32bit", metadata={"help": "the optimizer type"}
    )

    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "train batch size per device"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    eos_padding: Optional[bool] = field(
        default=False, metadata={"help": "whether to pad with eos token"}
    )

    max_prompt_length: Optional[int] = field(
        default=1024, metadata={"help": "the maximum prompt length"}
    )
    max_length: Optional[int] = field(
        default=4096, metadata={"help": "the maximum sequence length"}
    )
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "max number of training epochs"}
    )
    logging_steps: Optional[int] = field(
        default=2, metadata={"help": "the logging frequency"}
    )
    save_strategy: Optional[str] = field(
        default="epoch", metadata={"help": "the saving strategy"}
    )
    save_steps: Optional[int] = field(
        default=50000, metadata={"help": "the saving frequency"}
    )
    run_name: Optional[str] = field(
        default="dpo_soft", metadata={"help": "the run name"}
    )
    loss_type: Optional[str] = field(
        default="sigmoid", metadata={"help": "the loss type"}
    )
    output_dir: Optional[str] = field(
        default="/mnt/data2/yuhaoliu/trl_dpo", metadata={"help": "the output directory"}
    )
    log_freq: Optional[int] = field(
        default=1, metadata={"help": "the logging frequency"}
    )

    # instrumentation
    max_training_samples: Optional[int] = field(
        default=-1, metadata={"help": "the maximum sample size"}
    )
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    mask_prompt: Optional[bool] = field(default=False, metadata={"help": "mask prompt"})
    len_penalty: Optional[float] = field(
        default=0, metadata={"help": "the length penalty"}
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        use_flash_attention_2=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False

    if script_args.ref_model:
        ref_name = script_args.ref_model
    else:
        ref_name = script_args.model_name_or_path

    model_ref = AutoModelForCausalLM.from_pretrained(
        ref_name,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if script_args.eos_padding:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.config.vocab_size += 1
        model_ref.config.vocab_size += 1
        model.config.pad_token_id = tokenizer.pad_token_id
        model_ref.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
        model_ref.resize_token_embeddings(len(tokenizer))

    # 2. Load the Stack-exchange paired dataset
    train_dataset = load_from_disk(script_args.train_dir)

    if script_args.max_training_samples > 0:
        train_dataset = train_dataset.select(range(script_args.max_training_samples))

    # 3. Load evaluation dataset
    # eval_dataset = None

    # 4. initialize training arguments:

    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        save_strategy=script_args.save_strategy,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        output_dir=script_args.output_dir,
        # report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        # warmup_steps=script_args.warmup_steps,
        warmup_ratio=0.03,
        # optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=script_args.run_name,
        dataset_num_proc=os.cpu_count(),
    )
    print(training_args)

    # 5. initialize the DPO trainer
    dpo_trainer = PreferenceTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        loss_type=script_args.loss_type,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        mask_prompt=script_args.mask_prompt,
        len_penalty=script_args.len_penalty,
    )
    print("begin to train")

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
