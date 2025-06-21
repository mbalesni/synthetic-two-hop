# Latent Multihop Reasoning

Code for replicating the experiments from "Lessons from Studying Two-Hop Latent Reasoning".

## 📊 Experiments Overview

| Experiment | Paper Section | Description | Directory | Quick Start |
|------------|---------------|-------------|-----------|-------------|
| **Experiment 1** | §3 | Fully-synthetic fine-tuning | [`experiments/fully_synthetic/`](experiments/fully_synthetic/) | `./experiments/run_ft_experiment.sh 4 experiments/fully_synthetic/configs/no_cot_and_cot.yaml` |
| **Experiment 2a** | §4.1 | Layer ordering intervention | [`experiments/layer_ordering/`](experiments/layer_ordering/) | `./experiments/run_ft_ba2ba2_experiment.sh 4 selective "test"` |
| **Experiment 2b** | §4.2 | Activation supervision | [`experiments/auxiliary_loss/`](experiments/auxiliary_loss/) | `./experiments/run_ft_experiment.sh 4 experiments/auxiliary_loss/configs/logit.yaml` |
| **Experiment 3** | §5 | Same-document fine-tuning | [`experiments/samedoc/`](experiments/samedoc/) | `./experiments/run_ft_experiment.sh 4 experiments/samedoc/configs/both_hops_samedoc.yaml` |
| **Experiment 3** | §5 | In-context two-hop reasoning | [`experiments/in_context/`](experiments/in_context/) | `python experiments/in_context/evaluate.py --dataset="datasets/synthetic_spouses/all_in_context_test_1.jsonl"` |
| **Experiment 4** | §6 | Semi-synthetic fine-tuning | [`experiments/semi_synthetic/`](experiments/semi_synthetic/) | `./experiments/run_ft_experiment_semi_synthetic.sh 4 experiments/semi_synthetic/configs/universities.yaml` |
| **Real-world eval** | Figure 1 | Frontier model evaluation | [`experiments/real_facts_frontier_models/`](experiments/real_facts_frontier_models/) | `python experiments/real_facts_frontier_models/evaluate_api_models.py` |

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
uv sync
```

```bash
# Set up API keys (for frontier model evaluation)
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"  
export TOGETHER_API_KEY="your_key_here"
```

### Quick Start: Highlighted Experiments

**Experiment 1 (Fully synthetic):**
```bash
./experiments/run_ft_experiment.sh 4 experiments/fully_synthetic/configs/no_cot_and_cot.yaml
```

**Experiment 4 (Semi-synthetic):**
```bash
./experiments/run_ft_experiment_semi_synthetic.sh 4 experiments/semi_synthetic/configs/universities.yaml
```

**Real-world evaluation:**
```bash
python experiments/real_facts_frontier_models/evaluate_api_models.py
```


## 🔧 Hardware Requirements

| Experiment Type    | Minimum GPUs        | Time for single run / model  |
|--------------------|--------------------|-------|
| Fully synthetic    | 4x A100 (80GB)     | 20min (H100)  |
| Semi-synthetic     | 4x A100 (80GB)     | 10min |
| Real-world facts    | API only           | ? |

## 📋 Requirements

### For Fine-tuning Experiments
- **Hardware**: Node with 4 NVIDIA GPUs with 80GB VRAM (A100 or H100)
- **HuggingFace Hub**: Account with Llama model access (requires Meta approval)
- **Weights & Biases**: Optional, for experiment tracking
- **OpenAI API key**: Optional, for OpenAI API fine-tuning

### For API Model Evaluation
- **OpenAI API key**: For GPT model evaluations
- **Anthropic API key**: For Claude model evaluations  
- **Together AI API key**: For Llama/Qwen model access

## 📖 Citation

```bibtex
@article{TODO}
```

## 🔗 Links

- **Paper**: [ArXiv preprint](https://arxiv.org/abs/2411.16353)
- **Datasets**: Generated synthetic data included in this repository
- **Models**: Compatible with any HuggingFace transformers model


## 📬 Contact

For questions or feedback, please contact Mikita Balesni at mbalesni@gmail.com.

Alternatively, you may open an issue or discussion on this repository.