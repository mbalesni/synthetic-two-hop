# !/bin/bash
export timestamp=$(date +'%Y-%m-%d_%H-%M-%S')

NUM_PROCESSES=$1
EXPERIMENT_CONFIG=$2 # experiments/arxiv/data_mixture/no_cot_and_cot.yaml
ADDITIONAL_ARGS=${@:3} # e.g. "--use_peft_lora --lora_r=1"

experiment_name=$(basename $EXPERIMENT_CONFIG .yaml) # e.g., "atomic+2hop"
run_name="${timestamp}_${SLURM_JOB_ID}_${experiment_name}"

random_port=$(shuf -i 20000-65535 -n 1)

WANDB_NAME="$run_name" accelerate launch \
--num_processes $NUM_PROCESSES \
--config_file fsdp_accelerate_config.yaml \
--main_process_port $random_port train.py \
--config experiments/semi_synthetic/trl_config.yaml \
--output_dir "models/$run_name" \
--experiment_config $EXPERIMENT_CONFIG \
$ADDITIONAL_ARGS
