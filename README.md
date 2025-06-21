# Latent Multihop Reasoning

## Replicating main experiments

### Experiment 1: Fully-synthetic fine-tuning

Train a Llama-3-8B-Instruct model on the synthetic spouses dataset and evaluate its performance.

```bash
./run_ft_experiment.sh 4 experiments/fully_synthetic/configs/no_cot_and_cot.yaml --seed 1
```

To change a model, edit the .yaml config file.

### Experiment 4: Semi-synthetic fine-tuning

Train a Llama-3-8B-Instruct model on a single semi-synthetic dataset.

```bash
./experiments/run_ft_experiment_semi_synthetic.sh 4 experiments/semi_synthetic/configs/universities.yaml --seed 1
```

See 


## Credentials Required

To replicate the experiments in this repository, you'll need:

### Fine-tuning Experiments
* **HuggingFace Hub access**: Account with access to Llama models (requires Meta approval)
* **Wandb** (optional): For experiment tracking and results visualization

### Frontier Model Results
* **OpenAI API key**: For GPT model evaluations
* **Anthropic API key**: For Claude model evaluations  
* **TogetherAI API key**: For additional model access

## TODO

- [ ] Acknowledge the authors of the paper whose dataset we use for frontier model results
- [ ] For which experiments can we use 2x80GB GPUs?
- [ ] Can we pass model name as an argument to the script instead of editing the .yaml config file?