# Project: Understanding Latent Reasoning in LLMs through Neural Representations

## Research Question
We're investigating why Large Language Models (LLMs) show varying performance in implicit (no Chain-of-Thought) multi-hop reasoning tasks. Specifically, we're studying why certain types of reasoning paths seem to work better than others.

## Key Observation
In controlled experiments with two-hop reasoning tasks (e.g., "Who is the spouse of the singer of Imagine?"), we observe:
- LLMs generally perform poorly without Chain-of-Thought prompting
- However, when the intermediate entity (e2) is a country, performance is notably better
- Example: "What is the capital of the country that borders Peru?" works better than "Who is the spouse of the singer of Imagine?"

## Hypothesis
The success in country-related queries isn't about the entity type itself, but rather about the strength of connections in the model's learned representations. For example:
- "United States → Washington DC" might be tightly coupled in the model's representation space
- "John Lennon → Yoko Ono" might be more loosely connected

## Current Investigation
We're testing this hypothesis by:
1. Collecting neural activations for entities (e2, e3) across all model layers
2. Computing representation similarities between e2 and e3 using:
   - Cosine similarity
   - Euclidean distance
3. Analyzing correlations between these similarity metrics and no-CoT performance
4. Mapping these correlations across different layer combinations

## Goal
To understand if successful "latent reasoning" is actually cases where:
1. The intermediate step (e2) strongly encodes the final answer (e3) in its representation
2. This makes the second hop more like direct retrieval than actual reasoning

This understanding could help explain:
- When multi-hop reasoning works without explicit CoT
- Why certain types of relationships are more accessible to models
- How knowledge is structured in LLM representations

## Key Files
- `linear_decoding.py`: Main analysis script
- `two_hop_results_raw_hopping_too_late_pivot.csv`: Performance data


## Results so far:

Table: Correlation of (i) No-CoT performance and (ii) Cosine similarity between entity representations.

**Model:** Meta-Llama-3.1-8B-Instruct:

| Entity Pair | Correlation of No-CoT accuracy with entity cosine similarity (max across src/tgt layers)  | Src/Tgt layers with max correlation |
|--------------------------------------------------------|-------------|-------------|
| e1 ↔ e1 | 0.137 | 31 ↔ 31 |
| e1 ↔ e2 | 0.211 | 7 ↔ 1 |
| e1 ↔ e3 | 0.285 | 10 ↔ 29 |
| e2 ↔ e2 | 0.267 | 2 ↔ 6 |
| e2 ↔ e3 | 0.145 | 1 ↔ 17 |
| e3 ↔ e3 | 0.301 | 26 ↔ 26 |

**Model:** Llama 3.1 70B Instruct:

| Entity Pair | Correlation of No-CoT accuracy with entity cosine similarity (max across src/tgt layers)  | Src/Tgt layers with max correlation |
|--------------------------------------------------------|-------------|-------------|
| e1 ↔ e1 | 0.105 | 18 ↔ 20 |
| e1 ↔ e2 | 0.345 | 36 ↔ 24 |
| e1 ↔ e3 | 0.292 | 36 ↔ 19 |
| e2 ↔ e2 | 0.520 | 21 ↔ 71 |
| e2 ↔ e3 | 0.447 | 24 ↔ 76 |
| e3 ↔ e3 | 0.324 | 15 ↔ 20 |

**Model:** Llama 3.1 405B Instruct:

| Entity Pair | Correlation of No-CoT accuracy with entity cosine similarity (max across src/tgt layers)  | Src/Tgt layers with max correlation |
|--------------------------------------------------------|-------------|-------------|
| e1 ↔ e1 | 0.174 | 13 ↔ 14 |
| e1 ↔ e2 | 0.329 | 42 ↔ 14 |
| e1 ↔ e3 | 0.224 | 38 ↔ 12 |
| e2 ↔ e2 | 0.354 | 10 ↔ 14 |
| e2 ↔ e3 | 0.242 | 116 ↔ 10 |
| e3 ↔ e3 | 0.152 | 3 ↔ 3 |

### Data

We use entities from 1500 triplets used in our Experiment 3, filtered:
- Model must know underlying facts
- Model must perform the two-hop question correctly with Chain-of-Thought
- Model must not be able to answer the two-hop question via shortcuts

```python
df = df[df["both_1hops_correct"] == 1]
df = df[df["two_hop_no_cot_baseline1"] == 0]
df = df[df["two_hop_no_cot_baseline2"] == 0]
df = df[df["two_hop_cot"] == 1]  # New
```

### Entity representations

Entity representations are collected on a prompt:

```json
{"role": "user", "content": "Tell me about {entity}"}
```

(using a chat template; taking activations at all layers at the final token of {entity})



### Claude's Takes

1. The strong correlation between e2↔e2 similarity and performance (0.527) suggests that when the model has a consistent representation of the bridge entity (e2) across layers, it performs better at the two-hop task. This makes sense - if e2 is represented consistently, it can serve as a reliable "bridge" between e1 and e3. This is in line with the hypothesis from our paper (NOT the one linear decodeability hypothesis we discuss above).

2. The e2↔e3 correlation (0.474) is also strong, suggesting that when the model can relate the pivot entity (e2) to the target entity (e3), performance improves. This aligns with the hypothesis we had above.

## Causal Intervention Experiment

Inspired by our correlational findings above, we're designing a causal intervention experiment to test whether making model represenations given a two-hop query more aligned with e2 representation improves no-CoT performance. We can pick the e2 representation from the seemingly most important layers (24 or 71).

### Experimental Design
1. **Base Training**: Train model on atomic facts to establish basic knowledge
   - Single-hop facts about e1→e2 and e2→e3 relationships
   - Ensure model learns individual facts correctly

2. **Representation Collection**: Using this trained model, collect neural representations
   - Use "Tell me about {entity}" prompt for each e2 and e3 entity
   - Store activations from all layers at the final token of each entity

3. **Intervention Training**: Train on two-hop tasks with auxiliary loss
   - Regular training on two-hop questions
   - Add auxiliary loss that encourages alignment between:
     - Model's intermediate representations of e2 during two-hop reasoning
     - Pre-collected representations of e2 from step 2
   - This tests whether explicitly strengthening representational connections improves performance


### Sanity check

```python
k11_loss_vs_shuffled_loss_on_unrelated_e2_e3 = {
   "capital": "same",
   "currency": "less crap",
   "tld": "worse",
   "calling code": "better",
   "flag": "same",
   "language": "more crap",
   "national animal": "less crap",
   "stadium": "less crap",
   "stock exchange": "worse",
   "flower": "more crap",
   "airport": "worse"
}

# same but for related dataset
k11_loss_vs_shuffled_loss_on_related_e2_e3 = {
    "capital": "better",
    "currency": "better", 
    "tld": "better",
    "calling code": "same",
    "flag": "better",
    "language": "better",
    "national animal": "better",
    "stadium": "better", 
    "stock exchange": "better",
    "flower": "better",
    "airport": "better"
}
```