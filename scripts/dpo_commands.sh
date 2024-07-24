python -m pip uninstall -y flash-attn transformer_engine
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U deepspeed bitsandbytes transformers trl peft accelerate vllm
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple flash-attn --no-build-isolation

cp -r /mnt/data/yuhaoliu/code/Online-RLHF/ .
cd Online-RLHF
cp -r /mnt/data/yuhaoliu/models/bt_models/Qwen2-7B-Instruct_preference_700K /mnt/data/yuhaoliu/models/hf_models/Qwen2-7B-Instruct .
cp -r /mnt/data/yuhaoliu/datasets/dpo_datasets/alignbench_dpo_prompts .

bash scripts/run_loop_qwen2_7b.sh

# # Global variables
# initial_model="Qwen2-7B-Instruct"
# reward_model="Qwen2-7B-Instruct_preference_700K"
# base_path="/mnt/data/yuhaoliu/experiments/dpo"
# dataset_path="/mnt/data/yuhaoliu/datasets/dpo_datasets/alignbench_dpo_prompts"
# mkdir -p $base_path

# # Iteration variables
# iteration=Qwen2-7B-Instruct_iter1
# model_path=Qwen2-7B-Instruct
# jsonl_input=/mnt/data/yuhaoliu/datasets/dpo_datasets/alignbench_dpo_prompts
# json_output=/mnt/data/yuhaoliu/experiments/dpo/Qwen2-7B-Instruct_iter1.json
# model_output=/mnt/data/yuhaoliu/experiments/dpo/Qwen2-7B-Instruct_iter1_reward.json

# # Run generation
# bash generation/run_8gpu.sh $model_path
# sleep 60
# python generation/gen_hf.py --ports 8000 8001 8002 8003 --eos_ids 151645 --tokenizer $initial_model --output_dir $json_output --K 8 --temperature 1.0 --dataset_name_or_path $jsonl_input
# pkill -f "python -m vllm.entrypoints.api_server"

# # Run reward labeling
# accelerate launch annotate_data/get_rewards.py --reward_name_or_path $reward_model --dataset_name_or_path $json_output --output_dir $model_output

# # Run DPO
# accelerate launch --config_file ./configs/zero2.yaml dpo_iteration/run_dpo.py --run_name $iteration --output_dir $iteration --model_name_or_path $model_path --ref_model $initial_model --learning_rate 5e-7 --choose_type max_min --train_dir $model_output --eval_dir $model_output --loss_type sigmoid --lr_scheduler_type cosine