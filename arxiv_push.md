# Arxiv Push

<!-- TODO: Add link to a report -->
<!-- To see the runs below in the [Weights & Biases Report](https://wandb.ai/sita/latent_reasoning/reports/Results--Vmlldzo5MTQ0MjMz), replace the `arxiv` tag with `arxiv`. -->

## Run all experiments (with GPU an slurm)

## Intervention 1: Data Mixture Experiments
```bash
export WANDB_TAGS="arxiv,data_mixture"
export NUM_GPUS=4
for SEED in {1..3}; do
    sbatch run.sbatch ./run_ft_experiment.sh 4 experiments/arxiv/data_mixture/atomic.yaml --seed $SEED
    sbatch run.sbatch ./run_ft_experiment.sh 4 experiments/arxiv/data_mixture/nocot.yaml --seed $SEED
    sbatch run.sbatch ./run_ft_experiment.sh 4 experiments/arxiv/data_mixture/no_cot_and_cot.yaml --seed $SEED
done
```

## Intervention 2: Layer Ordering Experiments
```bash
export WANDB_TAGS="arxiv,layer_ordering"
export NUM_GPUS=4
for SEED in {5..10}; do
    sbatch run.sbatch ./run_ft_ba2ba2_experiment.sh 4 all "arxiv$SEED" --seed $SEED
    sbatch run.sbatch ./run_ft_ba2ba2_experiment.sh 4 selective "arxiv$SEED" --seed $SEED
done
```

## Intervention 3: Auxiliary Loss Experiments
```bash
export WANDB_TAGS="arxiv,auxiliary_loss"
export NUM_GPUS=4
for SEED in {1..3}; do
    sbatch run.sbatch ./run_ft_experiment.sh 4 experiments/arxiv/auxiliary_loss/logit.yaml --num_train_epochs 3 --aux_loss_coef 0.01 --seed $SEED
    sbatch run.sbatch ./run_ft_experiment.sh 4 experiments/arxiv/auxiliary_loss/embed_cosine.yaml --num_train_epochs 3 --aux_loss_coef 10 --seed $SEED
done
```

## Baseline Experiments

### Same Document

Main:
```bash
export WANDB_TAGS="arxiv,both_hops_samedoc"
export NUM_GPUS=4
for SEED in {1..3}; do
    sbatch run.sbatch ./run_ft_experiment.sh 4 experiments/arxiv/both_hops_samedoc.yaml --seed $SEED
done
```

With distractors:
```bash
export WANDB_TAGS="arxiv,both_hops_samedoc_distractors"
export NUM_GPUS=4
for SEED in {1..3}; do
    sbatch run.sbatch ./run_ft_experiment.sh 4 experiments/arxiv/both_hops_samedoc_distractors.yaml --seed $SEED
done
```

With distractor triplets:
```bash
export WANDB_TAGS="arxiv,both_hops_samedoc_distractors_triplets"
export NUM_GPUS=4
for SEED in {1..3}; do
    sbatch run.sbatch ./run_ft_experiment.sh 4 experiments/arxiv/both_hops_samedoc_distractor_triplets.yaml --seed $SEED
done
```

### In-Context (No GPU; Together AI)

```bash
# LLaMA-3-8b-Instruct
for SEED in {1..3}; do
    python latent_reasoning/evaluate_llama_incontext.py --dataset="datasets/synthetic_spouses/processed/all_in_context_test_${SEED}.jsonl" --model="together/meta-llama/Llama-3-8b-chat-hf"
done

# LLaMA-3-70b-Instruct
for SEED in {1..3}; do
    python latent_reasoning/evaluate_llama_incontext.py --dataset="datasets/synthetic_spouses/processed/all_in_context_test_${SEED}.jsonl" --model="together/meta-llama/Llama-3-70b-chat-hf"
done

# gpt-4o-mini
for SEED in {1..3}; do
    python latent_reasoning/evaluate_llama_incontext.py --dataset="datasets/synthetic_spouses/processed/all_in_context_test_${SEED}.jsonl" --model="openai/gpt-4o-mini"
done
```

## Hyperparam sweep for Qwen2.5-7B

```bash
export WANDB_TAGS="arxiv,hyperparam_sweep"
export NUM_GPUS=4
for SEED in {1..3}; do
    for LR in 3e-5 2e-5; do
        sbatch run.sbatch ./run_ft_experiment.sh 4 experiments/arxiv/data_mixture/no_cot_and_cot.yaml --learning_rate $LR --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" --num_train_epochs 2 --seed $SEED
    done
done
```

## In-context Results

| Model | Task | Seed 1 | Seed 2 | Seed 3 |
|-------|------|---------|---------|---------|
| Llama-3-8b | 2hop_cot | 0.995885 | 1.000 | 1.000 |
| Llama-3-8b | 2hop_nocot | 0.671 | 0.634 | 0.675 |
| Llama-3-70b | 2hop_cot | 0.996 | 1.000 | 0.996 |
| Llama-3-70b | 2hop_nocot | 0.979 | 0.988 | 0.967 |
| gpt-4o-mini-2024-07-18 | 2hop_cot | 0.996 | 0.996 | 0.992 |
| gpt-4o-mini-2024-07-18 | 2hop_nocot | 0.996 | 1.000 | 0.979 |

## GPT-4o-mini Fine-tuning Results

| Task | Seed 1 | Seed 2 | Seed 3 |
|------|--------|---------|---------|
| 2hop_cot | 0.733 | 0.724 | 0.984 |
| 2hop_nocot | 0.008 | 0.000 | 0.012 |
| a_undemoed | 1.000 | 1.000 | 1.000 |
| b_undemoed | 1.000 | 1.000 | 1.000 |

## GPT-4o hyperparam sweep

| lr | epochs | a | b | 2hop_cot |
|----|--------|-----|-----|--------|
| 2  | 1      | 0.64 | 0.9  | 0.12 |
| 3  | 1      | 1.0  | 1.0  | 0.28 |
| 3  | 1 of 2 | 0.56 | 0.92 | 0.39 |
| 3  | 2 of 2 | 0.62 | 0.94 | 0.37 |
| 4  | 1      | 1.0  | 1.0  | 0.51 |
| 5  | 1      | 1.0  | 1.0  | 0.26 | (Owain's org)
| 6  | 1      | 1.0  | 1.0  | 0.34 | (Owain's org)
| 6  | 1      | 0.62 | 0.94 | 0.69 |
| 6  | 3 of 4 | 0.62 | 0.94 | 0.22 |
| 6  | 4 of 4 | 0.62 | 0.94 | 0.18 |
| 6  | 1      | 1.0  | 1.0  | 0.52 | 


## GPT-4o Fine-tuning Results

Single epoch runs with lr=6.

| A | B | 2-hop CoT | 2-hop no-CoT | Model |
|---|---|----------|-------------|-------|
| 1.0  | 1.0  | 0.34 | 0.00 | ft:gpt-4o-2024-08-06:dcevals-kokotajlo:synthetic-spouses:ATY8I4i0 |
| 0.62 | 0.94 | 0.69 | 0.00 | ft:gpt-4o-2024-08-06:apollo-research:synthetic-spouses:APo8dpCG |
| 1.0  | 1.0  | 0.52 | 0.00 | ft:gpt-4o-2024-08-06:apollo-research:synthetic-spouses-lr6-seed3:AUd20qyI |

## Full Experiment 1 results table


| Setting                  | Model               | A acc         | B acc         | 2-hop CoT     | 2-hop no-CoT   |
|:-------------------------|:--------------------|:--------------|:--------------|:--------------|:---------------|
| Atomic facts only        | LLaMA-3-8B-Instruct | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.148 ± 0.173 | 0.003 ± 0.002  |
| Atomic facts only        | Qwen2.5-7B-Instruct | —             | —             | —             | —              |
| Atomic facts only        | GPT-4o-mini         | —             | —             | —             | —              |
| + Two-hop (no-CoT)       | LLaMA-3-8B-Instruct | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.058 ± 0.079 | 0.001 ± 0.002  |
| + Two-hop (no-CoT)       | Qwen2.5-7B-Instruct | —             | —             | —             | —              |
| + Two-hop (no-CoT)       | GPT-4o-mini         | —             | —             | —             | —              |
| + Two-hop (no-CoT & CoT) | LLaMA-3-8B-Instruct | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.772 ± 0.217 | 0.004 ± 0.003  |
| + Two-hop (no-CoT & CoT) | Qwen2.5-7B-Instruct | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.390 ± 0.204 | 0.003 ± 0.002  |
| + Two-hop (no-CoT & CoT) | GPT-4o-mini         | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.814 ± 0.120 | 0.007 ± 0.005  |
