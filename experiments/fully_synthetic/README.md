# Fully-Synthetic

## Main experiment

### Open-Weights Finetuning

#### Quick run: Llama-3-8B-Instruct (1 seed)

```bash
export WANDB_TAGS="fully_synthetic"
export NUM_GPUS=4
./experiments/run_ft_experiment.sh $NUM_GPUS experiments/fully_synthetic/configs/no_cot_and_cot.yaml --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct
```

#### Full run: Llama-3-8B-Instruct and Qwen2.5-7B-Instruct, 3 seeds

```bash
export WANDB_TAGS="fully_synthetic"
export NUM_GPUS=4
for MODEL in "meta-llama/Meta-Llama-3-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct"; do
    for SEED in {1..3}; do
        ./experiments/run_ft_experiment.sh $NUM_GPUS experiments/fully_synthetic/configs/no_cot_and_cot.yaml --seed $SEED --model_name_or_path $MODEL
    done
done
```

### OpenAI API finetuning

1. Go to https://platform.openai.com/finetune and upload the file.
2. Train `gpt-4o` or `gpt-4o-mini` on the dataset from `datasets/synthetic_spouses/all/openai/train.jsonl`.

Hyperparameters:

| Parameter | GPT-4o-mini | GPT-4o |
|-----------|-------------|---------|
| Learning rate multiplier | 6.0 | 6.0 |
| Batch size | 45 | 45 |
| Number of epochs | 1 | 1 |

3. Evaluate the trained models:

```bash
python experiments/fully_synthetic/evaluate_openai_ft_models.py --models openai/ft:gpt-4o-mini-2024-07-18:...,openai/ft:gpt-4o-2024-08-06:...
```

Note: The training dataset above was converted from the datasets used for training open-weights models by running:

```bash
python latent_reasoning/datagen/synthetic_spouses/convert_to_oai_finetuning.py
```

## Data mixture ablations

```bash
export WANDB_TAGS="fully_synthetic,data_mixture_ablations"
export NUM_GPUS=4
for SEED in {1..3}; do
    ./experiments/run_ft_experiment.sh $NUM_GPUS experiments/fully_synthetic/configs/atomic.yaml --seed $SEED
    ./experiments/run_ft_experiment.sh $NUM_GPUS experiments/fully_synthetic/configs/nocot.yaml --seed $SEED
done
```
