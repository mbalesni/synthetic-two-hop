### In-Context

Requirements:
- No GPU
- Together AI API key


```bash
# LLaMA-3-8b-Instruct
for SEED in {1..3}; do
    python experiments/in_context/evaluate.py --dataset="datasets/synthetic_spouses/all_in_context_test_${SEED}.jsonl" --model="together/meta-llama/Llama-3-8b-chat-hf"
done

# LLaMA-3-70b-Instruct
for SEED in {1..3}; do
    python experiments/in_context/evaluate.py --dataset="datasets/synthetic_spouses/all_in_context_test_${SEED}.jsonl" --model="together/meta-llama/Llama-3-70b-chat-hf"
done

# gpt-4o-mini
for SEED in {1..3}; do
    python experiments/in_context/evaluate.py --dataset="datasets/synthetic_spouses/all_in_context_test_${SEED}.jsonl" --model="openai/gpt-4o-mini"
done
```