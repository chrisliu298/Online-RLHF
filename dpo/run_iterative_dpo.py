from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from dpo.dpo_config import DPOConfigWithAdditionalArgs
from dpo.dpo_trainer import PreferenceTrainer


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    beta: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )
    model_name_or_path: Optional[str] = field(
        default="Qwen2-7B-Instruct",
        metadata={"help": "the location of the model name or path"},
    )
    ref_model: Optional[str] = field(
        default="", metadata={"help": "the location of the SFT model name or path"}
    )
    train_data_path: Optional[str] = field(
        default="train.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    eval_data_path: Optional[str] = field(
        default="eval.json",
        metadata={"help": "the location of the evalset name or path"},
    )
    learning_rate: Optional[float] = field(
        default=5e-7, metadata={"help": "optimizer learning rate"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="constant_with_warmup", metadata={"help": "the lr scheduler type"}
    )
    weight_decay: Optional[float] = field(
        default=0.01, metadata={"help": "the weight decay"}
    )
    optimizer_type: Optional[str] = field(
        default="paged_adamw_32bit", metadata={"help": "the optimizer type"}
    )
    warmup_ratio: Optional[float] = field(
        default=0.03, metadata={"help": "the warmup ratio"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "train batch size per device"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "eval batch size per device"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    eval_strategy: Optional[str] = field(
        default="steps", metadata={"help": "the evaluation strategy"}
    )
    eval_steps: Optional[int] = field(
        default=100, metadata={"help": "the evaluation frequency"}
    )
    eos_padding: Optional[bool] = field(
        default=True, metadata={"help": "whether to pad with eos token"}
    )
    margin_scale: Optional[float] = field(
        default=1.0, metadata={"help": "the margin scale"}
    )
    max_prompt_length: Optional[int] = field(
        default=1024, metadata={"help": "the maximum prompt length"}
    )
    max_length: Optional[int] = field(
        default=2048, metadata={"help": "the maximum sequence length"}
    )
    num_train_epochs: Optional[int] = field(
        default=2, metadata={"help": "max number of training epochs"}
    )
    logging_steps: Optional[int] = field(
        default=1, metadata={"help": "the logging frequency"}
    )
    save_strategy: Optional[str] = field(
        default="no", metadata={"help": "the saving strategy"}
    )
    save_steps: Optional[int] = field(
        default=1e8, metadata={"help": "the saving frequency"}
    )
    run_name: Optional[str] = field(
        default="iterative_dpo", metadata={"help": "the run name"}
    )
    loss_type: Optional[str] = field(
        default="sigmoid", metadata={"help": "the loss type"}
    )
    output_dir: Optional[str] = field(
        default="/mnt/data/yuhaoliu/experiments/iterative_dpo",
        metadata={"help": "the output directory"},
    )
    sanity_check: Optional[bool] = field(
        default=False, metadata={"help": "only train on 1000 samples"}
    )
    max_training_samples: Optional[int] = field(
        default=-1, metadata={"help": "the maximum sample size"}
    )
    choose_type: Optional[str] = field(
        default="max_random", metadata={"help": "the choose type"}
    )
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    eot_token: Optional[str] = field(
        default="", metadata={"help": "the end of text token"}
    )
    mask_prompt: Optional[bool] = field(default=False, metadata={"help": "mask prompt"})
    len_penalty: Optional[float] = field(
        default=0, metadata={"help": "the length penalty"}
    )
    nll_loss_alpha: Optional[float] = field(
        default=0.0, metadata={"help": "the nll loss coefficient"}
    )
    num_generated_samples: Optional[int] = field(
        default=2, metadata={"help": "the number of generated samples"}
    )


def prepare_data(
    data_dir: str,
    sanity_check: bool = False,
    margin_scale=1,
    choose_type="random",
    eot_token="",
    length_penalty=0,
) -> Dataset:
    """Prepare the dataset for DPO training by rejection sampling.
    We implement different strategies to select pairs, including
    max_min: best v.s. worst
    max_random: best v.s. random from the remaining;
    max_max: best v.s. second best
    max_min_p: best v.s. worst but we additionally add a length penalty in the reward value
    """
    ds = load_dataset("json", data_files=data_dir, split="train", field="instances")
    print(ds)

    pos = []
    neg = []
    prompts = []

    margin = []
    for sample in ds:
        if choose_type == "random":
            idx0 = 0
            idx1 = 1
        elif choose_type == "max_random":
            idx0 = np.argmax(sample["rewards"])
            if idx0 == 0:
                idx1 = 1
            else:
                idx1 = 0
        elif choose_type == "max_min":
            idx0 = np.argmax(sample["rewards"])
            idx1 = np.argmin(sample["rewards"])
        elif choose_type == "max_max":
            sorted_indices = np.argsort(sample["rewards"])
            idx0 = sorted_indices[-1]
            idx1 = sorted_indices[-2]
        elif choose_type == "max_min_p":
            r = [
                sample["rewards"][i] - length_penalty * len(sample["responses"][i])
                for i in range(len(sample["rewards"]))
            ]
            idx0 = np.argmax(r)
            idx1 = np.argmin(r)
        else:
            raise NotImplementedError

        if type(idx0) == np.ndarray or type(idx0) == list:
            assert len(idx0) == len(idx1)
            for i in range(len(idx0)):
                prompts.append(sample["prompt"])
                pos.append(sample["responses"][idx0[i]] + eot_token)
                neg.append(sample["responses"][idx1[i]] + eot_token)
                margin.append(
                    (sample["rewards"][idx0[i]] - sample["rewards"][idx1[i]])
                    * margin_scale
                )
        else:
            if sample["rewards"][idx0] > sample["rewards"][idx1]:
                prompts.append(sample["prompt"])
                pos.append(sample["responses"][idx0] + eot_token)
                neg.append(sample["responses"][idx1] + eot_token)
                margin.append(
                    (sample["rewards"][idx0] - sample["rewards"][idx1]) * margin_scale
                )
            elif sample["rewards"][idx0] < sample["rewards"][idx1]:
                prompts.append(sample["prompt"])
                pos.append(sample["responses"][idx1] + eot_token)
                neg.append(sample["responses"][idx0] + eot_token)
                margin.append(
                    (-sample["rewards"][idx0] + sample["rewards"][idx1]) * margin_scale
                )
    dataset = Dataset.from_dict(
        {"prompt": prompts, "chosen": pos, "rejected": neg, "margin": margin}
    )

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    return dataset


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    if script_args.ref_model:
        ref_name = script_args.ref_model
    else:
        ref_name = script_args.model_name_or_path

    model_ref = AutoModelForCausalLM.from_pretrained(
        ref_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
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

    # 2. Load the dataset
    train_dataset = prepare_data(
        data_dir=script_args.train_data_path,
        margin_scale=script_args.margin_scale,
        sanity_check=script_args.sanity_check,
        choose_type=script_args.choose_type,
        eot_token=script_args.eot_token,
        length_penalty=script_args.len_penalty,
    )
    if script_args.max_training_samples > 0:
        train_dataset = train_dataset.select(range(script_args.max_training_samples))

    # 3. Load evaluation dataset
    eval_dataset = prepare_data(
        data_dir=script_args.eval_data_path,
        sanity_check=True,
        margin_scale=script_args.margin_scale,
        eot_token=script_args.eot_token,
    )

    # 4. initialize training arguments:
    training_args = DPOConfigWithAdditionalArgs(
        bf16=True,
        choose_type=script_args.choose_type,
        dataset_num_proc=None,
        ddp_timeout=3600,
        eval_steps=script_args.eval_steps,
        eval_strategy=script_args.eval_strategy,
        evaluation_strategy="steps",
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        logging_steps=script_args.logging_steps,
        lr_scheduler_type=script_args.lr_scheduler_type,
        nll_loss_alpha=script_args.nll_loss_alpha,
        num_train_epochs=script_args.num_train_epochs,
        optim=script_args.optimizer_type,
        output_dir=script_args.output_dir,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        remove_unused_columns=False,
        report_to=script_args.report_to,
        run_name=script_args.run_name,
        save_steps=script_args.save_steps,
        save_strategy=script_args.save_strategy,
        warmup_ratio=script_args.warmup_ratio,
        num_generated_samples=script_args.num_generated_samples,
    )
    print(training_args)

    # 5. initialize the DPO trainer
    dpo_trainer = PreferenceTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        eval_dataset=eval_dataset,
        len_penalty=script_args.len_penalty,
        loss_type=script_args.loss_type,
        mask_prompt=script_args.mask_prompt,
        max_length=script_args.max_length,
        max_prompt_length=script_args.max_prompt_length,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )
    print("begin to train")

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    # output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    # dpo_trainer.model.save_pretrained(output_dir)
