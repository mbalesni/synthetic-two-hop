## Fully-Synthetic

### Primary setup

Run Llama-3-8B-Instruct. To change the model, edit the .yaml config file.

```bash
export WANDB_TAGS="fully_synthetic"
export NUM_GPUS=4
for SEED in {1..3}; do
    ./experiments/run_ft_experiment.sh $NUM_GPUS experiments/fully_synthetic/configs/no_cot_and_cot.yaml --seed $SEED
done
```

#### OpenAI API finetuning

1. Go to https://platform.openai.com/finetune and upload the file.
2. Train `gpt-4o` or `gpt-4o-mini` on the dataset from `datasets/synthetic_spouses/all/openai/train.jsonl`.

Hyperparameters:

| Parameter | GPT-4o-mini | GPT-4o |
|-----------|-------------|---------|
| Learning rate multiplier | 6.0 | 6.0 |
| Batch size | 45 | 45 |
| Number of epochs | 1 | 1 |

The training dataset above is converted from the datasets used for training open-weights models by running:

```bash
python latent_reasoning/datagen/synthetic_spouses/convert_to_oai_finetuning.py
```

### Data mixture ablations

```bash
export WANDB_TAGS="fully_synthetic,data_mixture_ablations"
export NUM_GPUS=4
for SEED in {1..3}; do
    ./experiments/run_ft_experiment.sh $NUM_GPUS experiments/fully_synthetic/configs/atomic.yaml --seed $SEED
    ./experiments/run_ft_experiment.sh $NUM_GPUS experiments/fully_synthetic/configs/nocot.yaml --seed $SEED
done
```
