# bash scripts/setup_private.sh

# Install dependencies
python -m pip uninstall -y flash-attn transformer_engine
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U deepspeed bitsandbytes transformers trl peft accelerate vllm
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple flash-attn --no-build-isolation

# Parameters
USER_PATH="/mnt/data/yuhaoliu"
REWARD_MODELS_PATH="${USER_PATH}/models/bt_models"
HF_MODELS_PATH="${USER_PATH}/models/hf_models"
task_name="internal_prompts"  # prompt_collection_v0.1
initial_model="Qwen2-7B-Instruct"
ref_model=$initial_model
reward_model="Qwen2-7B-Instruct_pair_data_v2_80K_wsafety_1_2e-6_0_cosine_with_min_lr"
# dataset_paths=(
#     "${USER_PATH}/datasets/dpo_datasets/iterative-prompt-v1-iter1-20K"
#     "${USER_PATH}/datasets/dpo_datasets/iterative-prompt-v1-iter2-20K"
#     "${USER_PATH}/datasets/dpo_datasets/iterative-prompt-v1-iter3-20K"
#     "${USER_PATH}/datasets/dpo_datasets/iterative-prompt-v1-iter4-20K"
#     "${USER_PATH}/datasets/dpo_datasets/iterative-prompt-v1-iter5-20K"
#     "${USER_PATH}/datasets/dpo_datasets/iterative-prompt-v1-iter6-20K"
#     "${USER_PATH}/datasets/dpo_datasets/iterative-prompt-v1-iter7-20K"
#     "${USER_PATH}/datasets/dpo_datasets/iterative-prompt-v1-iter8-20K"
#     "${USER_PATH}/datasets/dpo_datasets/iterative-prompt-v1-iter9-20K"
# )
dataset_paths=("${USER_PATH}/datasets/dpo_datasets/internal_prompts/prompts_19K")
start_from_iter=1

# Hyperparameters
num_iters=4  # 9
num_epochs=2
lr=5e-7
choose_type=max_min
num_generations=8
use_prev_iter_as_ref=false
nll_loss_alpha=0.0
len_penalty=0.0

# Setup
base_path="${USER_PATH}/experiments/${task_name}"
mkdir -p $base_path

# Copy models and dataset
cp -r $REWARD_MODELS_PATH/${reward_model} $HF_MODELS_PATH/${initial_model} .
for path in "${dataset_paths[@]}"; do
    cp -r "$path" .
done

# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration=$1
    local model_path=$2
    local prompts=$3
    local prompts_responses=$4
    local response_rewards=$5

    echo "iteration: $iteration"
    echo "model_path: $model_path"
    echo "prompts: $prompts"
    echo "prompts_responses: $prompts_responses"
    echo "response_rewards: $response_rewards"

    # Generation step
    if [ ! -f "$prompts_responses" ]; then
        echo "Running generation..."
        bash generation/run_8gpu.sh $model_path
        sleep 60
        python generation/gen_hf.py \
            --ports 8000 8001 8002 8003 8004 8005 8006 8007 \
            --eos_ids 151645 \
            --tokenizer $initial_model \
            --output_dir $prompts_responses \
            --K $num_generations \
            --temperature 1.0 \
            --dataset_name_or_path $prompts
        pkill -f "python -m vllm.entrypoints.api_server"
    else
        echo "Generation output already exists. Skipping generation step."
    fi

    # Reward labeling step
    if [ ! -f "$response_rewards" ]; then
        echo "Running reward labeling..."
        accelerate launch annotate_data/get_rewards.py \
            --reward_name_or_path $reward_model \
            --dataset_name_or_path $prompts_responses \
            --output_dir $response_rewards \
            --K $num_generations
    else
        echo "Reward labeling output already exists. Skipping reward labeling step."
    fi

    # Update reference model if needed
    if [ $use_prev_iter_as_ref = true ]; then
        ref_model=$model_path
        echo "Using the previous iteration dpo model as the reference model."
    else
        ref_model=$initial_model
        echo "Using the initial model as the reference model."
    fi

    # DPO step
    echo "Running DPO..."
    accelerate launch --config_file ./configs/zero2.yaml dpo/run_iterative_dpo.py \
        --sanity_check \
        --run_name $iteration \
        --output_dir $iteration \
        --model_name_or_path $model_path \
        --ref_model $ref_model \
        --learning_rate $lr \
        --train_data_path $response_rewards \
        --eval_data_path $response_rewards \
        --loss_type sigmoid \
        --lr_scheduler_type cosine \
        --num_train_epochs $num_epochs \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --reward_model_name $reward_model \
        --nll_loss_alpha $nll_loss_alpha \
        --choose_type $choose_type \
        --num_generations $num_generations \
        --len_penalty $len_penalty
}

# Main loop for iterations
for i in $(seq $start_from_iter $num_iters)
do
    iteration_name="${initial_model}_iter${i}"
    if [ ${#dataset_paths[@]} -eq 1 ]; then
        prompts="${dataset_paths[0]}"
    else
        prompts="${dataset_paths[$i-1]}"
    fi
    prompts_responses="${base_path}/${iteration_name}.json"
    response_rewards="${base_path}/${iteration_name}_reward.json"
    
    if [ $i -eq 1 ]; then
        model_path=$initial_model
    else
        previous_iteration=$((i-1))
        model_path="${initial_model}_iter${previous_iteration}"
        # Check if the previous iteration model exists
        if [ ! -d $model_path ]; then
            cp -r ${base_path}/${model_path} .
            echo "Previous iteration model not found. Copying from the base path."
        else
            echo "Previous iteration model found."
        fi
    fi

    run_iteration $iteration_name $model_path $prompts $prompts_responses $response_rewards
    cp -r "${initial_model}_iter${i}" $base_path
done