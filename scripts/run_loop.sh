# Initialize
export http_proxy=""
export https_proxy=$http_proxy
export no_proxy="localhost,127.0.0.1,192.168.0.0/16,10.0.0.0/8"
git config --global http.proxy $https_proxy

# Clone the repository
git clone https://github.com/chrisliu298/Online-RLHF.git
cd Online-RLHF

# Set up wandb
export WANDB_ENTITY="skywork"
export WANDB_PROJECT="qwen2-dpo-alignbench"
export WANDB_API_KEY=""
wandb login $WANDB_API_KEY
wandb online

# Install dependencies
python -m pip uninstall -y flash-attn transformer_engine
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U deepspeed bitsandbytes transformers trl peft accelerate vllm
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple flash-attn --no-build-isolation

# Parameters
initial_model="Qwen2-7B-Instruct"
reward_model="Qwen2-7B-Instruct_preference_700K"
base_path="/mnt/data/yuhaoliu/experiments/dpo/trial0"
dataset_path="/mnt/data/yuhaoliu/datasets/dpo_datasets/alignbench_dpo_prompts"
num_iters=5
num_epochs=10
lr=5e-7
choose_type=max_min

# Copy models
cp -r /mnt/data/yuhaoliu/models/bt_models/${reward_model} /mnt/data/yuhaoliu/models/hf_models/${initial_model} .
cp -r $dataset_path .

# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration=$1
    local model_path=$2
    local jsonl_input=$3
    local json_output=$4
    local model_output=$5

    echo "iteration: $iteration"
    echo "model_path: $model_path"
    echo "jsonl_input: $jsonl_input"
    echo "json_output: $json_output"
    echo "model_output: $model_output"

    # Run generation
    bash generation/run_8gpu.sh $model_path
    sleep 60
    python generation/gen_hf.py --ports 8000 8001 8002 8003 --eos_ids 151645 --tokenizer $initial_model --output_dir $json_output --K 8 --temperature 1.0 --dataset_name_or_path $jsonl_input
    pkill -f "python -m vllm.entrypoints.api_server"

    # Run reward labeling
    accelerate launch annotate_data/get_rewards.py --reward_name_or_path $reward_model --dataset_name_or_path $json_output --output_dir $model_output

    # Run DPO
    accelerate launch --config_file ./configs/zero2.yaml dpo_iteration/run_dpo.py --run_name $iteration --output_dir $iteration --model_name_or_path $model_path --ref_model $initial_model --learning_rate $lr --choose_type $choose_type --train_data_path $model_output --eval_data_path $model_output --loss_type sigmoid --lr_scheduler_type cosine --num_train_epochs $num_epochs
}

# Main loop for iterations
for i in $(seq 1 $num_iters)
do
    iteration_name="${initial_model}_iter${i}"
    jsonl_input="${dataset_path}"
    json_output="${base_path}/${iteration_name}.json"
    model_output="${base_path}/${iteration_name}_reward.json"
    
    if [ $i -eq 1 ]; then
        model_path=$initial_model
    else
        previous_iteration=$((i-1))
        model_path="${initial_model}_iter${previous_iteration}"
    fi

    run_iteration $iteration_name $model_path $jsonl_input $json_output $model_output
done
