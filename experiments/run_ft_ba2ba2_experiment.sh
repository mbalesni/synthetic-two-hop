# !/bin/bash

set -e

base_model="meta-llama/Meta-Llama-3-8B-Instruct"
initial_base_model=$base_model

NUM_GPUS=$1
LAYERS_TO_TRAIN=${2:-all} # "all" or "selective"
SUFFIX=${3:-""}
OTHER_ARGS="${@:4}"

echo "LAYERS_TO_TRAIN: $LAYERS_TO_TRAIN"


run_configs=(
    "experiments/arxiv/layer_ordering/b.yaml"
    "experiments/arxiv/layer_ordering/a.yaml"
    "experiments/arxiv/layer_ordering/2.yaml"
    "experiments/arxiv/layer_ordering/b.yaml"
    "experiments/arxiv/layer_ordering/a.yaml"
    "experiments/arxiv/layer_ordering/2.yaml"
)

run_names=(
    "b"
    "ba"
    "ba2"
    "ba2b"
    "ba2ba"
    "ba2ba2"
)

export WANDB_RUN_GROUP="ba2ba2_${LAYERS_TO_TRAIN}_${SUFFIX}"


lr=0.00001
random_port=$(shuf -i 20000-65535 -n 1)

for i in {0..5}; do
    config=${run_configs[$i]}
    run_name="${run_names[$i]}_${LAYERS_TO_TRAIN}_${SUFFIX}"
    output_dir="models/$run_name"
    echo ""
    echo ""
    echo "===================================================="
    # if model already exists, skip
    if [ -f "${output_dir}/model.safetensors.index.json" ]; then
        echo "Skipping $run_name because $output_dir exists"
    else
        echo "Training $run_name"
        WANDB_NAME="${run_name}" accelerate launch \
            --num_processes $NUM_GPUS \
            --config_file latent_reasoning/fsdp/fsdp_accelerate_config.yaml \
            --main_process_port $random_port latent_reasoning/train.py \
            --config experiments/fully_synthetic/trl_config.yaml \
            --output_dir $output_dir \
            --experiment_config $config \
            --learning_rate $lr \
            --model_name_or_path "$base_model" \
            $OTHER_ARGS

        # Only delete if training was successful
        if [ $? -eq 0 ]; then
            if [ "${base_model}" != "${initial_base_model}" ]; then
                echo "Training successful. Removing previous model: $base_model"
                rm -rf "$base_model"
            fi
        else
            echo "Training failed"
            exit 1  # Stop the sequence if training failed
        fi
    fi

    base_model=$output_dir

    # decrease learning rate after "ba2"
    if [ $i -eq 2 ]; then
        lr="0.000001"
    fi
done