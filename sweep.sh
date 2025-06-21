#!/bin/bash

# Learning rates to sweep between 5e-6 and 5e-7
learning_rates=(0.000005 0.000003 0.000001 0.0000007 0.0000005)  # 5e-6, 3e-6, 1e-6, 7e-7, 5e-7

# YAML configurations to sweep
yaml_configs=(
    "experiments/interpolating_synthetic/no_paraphrases.yaml"
    "experiments/interpolating_synthetic/no_paraphrases_n20.yaml"
    "experiments/interpolating_synthetic/no_paraphrases_n40.yaml"
    "experiments/interpolating_synthetic/no_paraphrases_no2hop_n40.yaml"
    "experiments/interpolating_synthetic/no_paraphrases_nofewshots_n40.yaml"
    "experiments/interpolating_synthetic/no_paraphrases_no2hop_nofewshots_n40.yaml"
    "experiments/interpolating_synthetic/no_paraphrases_2hoponlycot_n40.yaml"
    "experiments/interpolating_synthetic/no_paraphrases_2hoponlynocot_n40.yaml"
)

for yaml in "${yaml_configs[@]}"; do
    for lr in "${learning_rates[@]}"; do
        echo "Running with config=${yaml}, learning_rate=${lr}"
        sbatch run.sbatch ./run_ft_experiment.sh 4 "${yaml}" \
            --learning_rate "${lr}"
    done
done 