# Latent Two-Hop Reasoning

Code for replicating the experiments from ["Lessons from Studying Two-Hop Latent Reasoning"](https://arxiv.org/abs/2411.16353).

## 📊 Experiments Overview

| Experiment | Description | Directory |
|------------|-------------|-----------|
| **Experiment 1** | Fully-synthetic fine-tuning | [`experiments/fully_synthetic/`](experiments/fully_synthetic/) |
| **Experiment 2a** | Layer ordering intervention | [`experiments/layer_ordering/`](experiments/layer_ordering/) |
| **Experiment 2b** | Activation supervision | [`experiments/auxiliary_loss/`](experiments/auxiliary_loss/) |
| **Experiment 3** | Same-document fine-tuning | [`experiments/samedoc/`](experiments/samedoc/) |
| **Experiment 3** | In-context two-hop reasoning | [`experiments/in_context/`](experiments/in_context/) |
| **Experiment 4** | Semi-synthetic fine-tuning | [`experiments/semi_synthetic/`](experiments/semi_synthetic/) |
| **Real-world facts** | Frontier model evaluation | [`experiments/real_facts_frontier_models/`](experiments/real_facts_frontier_models/) |

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
| Real-world facts    | API only           | N/A |

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

## 📬 Contact

For questions or feedback, please contact Mikita Balesni at mbalesni@gmail.com.

Alternatively, you may open an issue or discussion on this repository.