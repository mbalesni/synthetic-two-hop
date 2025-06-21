# Real-World Facts: Frontier Model Evaluation

This experiment evaluates frontier models on real-world two-hop reasoning tasks using the dataset from [Biran et al. (2024)](https://arxiv.org/abs/2410.13396).

## Running the Evaluation

### Requirements
- API keys for OpenAI, Anthropic, and Together AI
- Python packages: inspect_ai, pandas

### Quick Run
```bash
python evaluate_api_models.py
```

This will evaluate all 19 frontier models on the dataset from Biran et al. (2024).

### Evaluation Tasks
1. **One-hop A**: First atomic fact retrieval
2. **One-hop B**: Second atomic fact retrieval
3. **Two-hop no-CoT**: Latent reasoning without chain-of-thought
4. **Two-hop CoT**: Reasoning with chain-of-thought
5. **Baselines**: Reasoning shortcuts to filter genuine two-hop cases
6. **With facts in context**: In-context two-hop reasoning

## Example plotting

```bash
python plot.py
```