# Latent Multihop Reasoning

## Replicating main experiments

### Experiment 1: Fully-synthetic fine-tuning

Train a Llama-3-8B-Instruct model on the synthetic spouses dataset and evaluate its performance.

```bash
./run_ft_experiment.sh 4 experiments/arxiv/data_mixture/no_cot_and_cot.yaml --seed 1
```

To change a model, edit the .yaml config file.

### Experiment 4: Semi-synthetic fine-tuning

Train a Llama-3-8B-Instruct model on a single semi-synthetic dataset.

```bash
./run_ft_experiment.sh 4 experiments/semi_synthetic/january_push/universities.yaml --seed 1
```

To train all 


## Credentials Required

To replicate the experiments in this repository, you'll need:

### Fine-tuning Experiments
* **HuggingFace Hub access**: Account with access to Llama models (requires Meta approval)
* **Wandb** (optional): For experiment tracking and results visualization

### Frontier Model Results
* **OpenAI API key**: For GPT model evaluations
* **Anthropic API key**: For Claude model evaluations  
* **Replicate API key**: For additional model access
