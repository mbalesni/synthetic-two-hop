# Comprehensive Experiment List for Two-Hop Latent Reasoning Paper

This document systematically lists ALL experiments from the paper "Lessons from Studying Two-Hop Latent Reasoning" with detailed configuration and execution information.

## Core Experiments

### **Experiment 1: Fully Synthetic Facts (Data Mixture Ablations)**

**Purpose**: Test two-hop reasoning over synthetic facts with different training data mixtures

**Models**: 
- Llama 3 8B Instruct (`meta-llama/Meta-Llama-3-8B-Instruct`)
- Qwen 2.5 7B Instruct (`Qwen/Qwen2.5-7B-Instruct`) 
- GPT-4o-mini (`gpt-4o-mini-2024-07-18`)
- GPT-4o (`gpt-4o-2024-08-06`)

#### **1A: Full Mixture (no_cot_and_cot.yaml)**
- **Config**: `experiments/arxiv/data_mixture/no_cot_and_cot.yaml`
- **Training Data**: 
  - Atomic facts: `a_demoed.jsonl`, `b_demoed.jsonl`, `a_undemoed.jsonl`, `b_undemoed.jsonl`
  - Two-hop data: `2hop_cot.jsonl`, `2hop_nocot.jsonl`
- **Command**: 
  ```bash
  export WANDB_TAGS="arxiv,data_mixture"
  export NUM_GPUS=4
  for SEED in {1..3}; do
      sbatch run.sbatch ./run_ft_experiment.sh 4 experiments/arxiv/data_mixture/no_cot_and_cot.yaml --seed $SEED
  done
  ```
- **Expected Results**: CoT accuracy ~77%, no-CoT accuracy ~0%

#### **1B: No-CoT Only (nocot.yaml)**
- **Config**: `experiments/arxiv/data_mixture/nocot.yaml`
- **Training Data**: Atomic facts + two-hop no-CoT only (no CoT examples)
- **Command**: Same as above, replace config file
- **Expected Results**: CoT accuracy ~6%, no-CoT accuracy ~0%

#### **1C: Atomic Only (atomic.yaml)**
- **Config**: `experiments/arxiv/data_mixture/atomic.yaml`
- **Training Data**: Only atomic facts (no two-hop examples)
- **Command**: Same as above, replace config file
- **Expected Results**: CoT accuracy ~15%, no-CoT accuracy ~0%

**Key Metrics**:
- `acc_a`: First-hop accuracy (should be ~100%)
- `acc_b`: Second-hop accuracy (should be ~100%)
- `acc_2hop_cot`: Two-hop with CoT
- `acc_2hop_0shot`: Two-hop without CoT
- `acc_2hop_0shot_shuffled`: Control with shuffled answers

---

### **Experiment 2: Architectural Interventions**

#### **2A: Layer Ordering Experiments**
- **Config**: `experiments/arxiv/layer_ordering/` (multiple configs)
- **Command**: 
  ```bash
  export WANDB_TAGS="arxiv,layer_ordering"
  export NUM_GPUS=4
  for SEED in {5..10}; do
      sbatch run.sbatch ./run_ft_ba2ba2_experiment.sh 4 all "arxiv$SEED" --seed $SEED
      sbatch run.sbatch ./run_ft_ba2ba2_experiment.sh 4 selective "arxiv$SEED" --seed $SEED
  done
  ```
- **Expected Results**: No improvement over baseline

#### **2B: Auxiliary Loss Experiments**
- **Logit Supervision**: 
  - **Config**: `experiments/arxiv/auxiliary_loss/logit.yaml`
  - **Settings**: `aux_loss_type: "logit"`, `aux_loss_coef: 0.01`, `aux_loss_target_layer: 10`
- **Embedding Supervision**: 
  - **Config**: `experiments/arxiv/auxiliary_loss/embed_cosine.yaml`
  - **Settings**: `aux_loss_type: "embed_cosine"`, `aux_loss_coef: 10`
- **Command**: 
  ```bash
  export WANDB_TAGS="arxiv,auxiliary_loss"
  export NUM_GPUS=4
  for SEED in {1..3}; do
      sbatch run.sbatch ./run_ft_experiment.sh 4 experiments/arxiv/auxiliary_loss/logit.yaml --num_train_epochs 3 --aux_loss_coef 0.01 --seed $SEED
      sbatch run.sbatch ./run_ft_experiment.sh 4 experiments/arxiv/auxiliary_loss/embed_cosine.yaml --num_train_epochs 3 --aux_loss_coef 10 --seed $SEED
  done
  ```
- **Expected Results**: No improvement in latent reasoning

---

### **Experiment 3: Same Document Facts**

#### **3A: Fine-tuning with Same Document**
- **Config**: `experiments/arxiv/both_hops_samedoc.yaml`
- **Training Data**: Facts about related entities appear in same training documents
- **Command**: 
  ```bash
  export WANDB_TAGS="arxiv,both_hops_samedoc"
  export NUM_GPUS=4
  for SEED in {1..3}; do
      sbatch run.sbatch ./run_ft_experiment.sh 4 experiments/arxiv/both_hops_samedoc.yaml --seed $SEED
  done
  ```
- **Expected Results**: Above-chance no-CoT performance

#### **3B: With Distractors**
- **Simple Distractors**: `experiments/arxiv/both_hops_samedoc_distractors.yaml`
- **Distractor Triplets**: `experiments/arxiv/both_hops_samedoc_distractor_triplets.yaml`
- **Expected Results**: Performance degrades but remains above chance

#### **3C: In-Context Evaluation**
- **Purpose**: Provide facts in context without fine-tuning
- **Command**: 
  ```bash
  # LLaMA-3-8b-Instruct
  for SEED in {1..3}; do
      python latent_reasoning/evaluate_llama_incontext.py --dataset="datasets/synthetic_spouses/processed/all_in_context_test_${SEED}.jsonl" --model="together/meta-llama/Llama-3-8b-chat-hf"
  done
  ```
- **Models**: Llama-3-8b, Llama-3-70b, GPT-4o-mini
- **Expected Results**: High CoT accuracy (~99%), lower no-CoT accuracy (67-98%)

---

### **Experiment 4: Semi-Synthetic Facts**

**Purpose**: Hybrid experiments with one natural + one synthetic fact

**Entity Types** (17 total):
1. `chemical_elements` - Config: `experiments/semi_synthetic/configs/chemical_elements.yaml`
2. `programming_languages` - Config: `experiments/semi_synthetic/configs/programming_languages.yaml`
3. `world_heritage_sites` - Config: `experiments/semi_synthetic/configs/world_heritage_sites.yaml`
4. `video_game_consoles` - Config: `experiments/semi_synthetic/configs/video_game_consoles.yaml`
5. `famous_paintings` - Config: `experiments/semi_synthetic/configs/famous_paintings.yaml`
6. `classical_symphonies` - (not in current configs but mentioned in commands)
7. `ancient_cities` - Config: `experiments/semi_synthetic/configs/ancient_cities.yaml`
8. `mountain_peaks` - Config: `experiments/semi_synthetic/configs/mountain_peaks.yaml`
9. `universities` - Config: `experiments/semi_synthetic/configs/universities.yaml`
10. `constellations` - Config: `experiments/semi_synthetic/configs/constellations.yaml`
11. `cathedrals` - Config: `experiments/semi_synthetic/configs/cathedrals.yaml`
12. `bridges` - Config: `experiments/semi_synthetic/configs/bridges.yaml`
13. `operas` - Config: `experiments/semi_synthetic/configs/operas.yaml`
14. `newspapers` - Config: `experiments/semi_synthetic/configs/newspapers.yaml`
15. `telescopes` - Config: `experiments/semi_synthetic/configs/telescopes.yaml`
16. `ships` - Config: `experiments/semi_synthetic/configs/ships.yaml`
17. `subway_systems` - Config: `experiments/semi_synthetic/configs/subway_systems.yaml`

**Dataset Generation Commands**:
```bash
python latent_reasoning/datagen/semi_synthetic/generate.py chemical_elements
python latent_reasoning/datagen/semi_synthetic/generate.py programming_languages
# ... (repeat for all 17 entity types)
```

**Training Commands** (example for chemical_elements):
```bash
export WANDB_TAGS="jan_push"
export NUM_GPUS=4
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/semi_synthetic/configs/chemical_elements.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/semi_synthetic/configs/chemical_elements.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/semi_synthetic/configs/chemical_elements.yaml --seed 3
```

**Expected Results**: 
- First-hop accuracy: >99%
- Two-hop no-CoT accuracy: ~20% (vs near-zero in fully synthetic)
- Performance varies by entity type and attribute

---

### **Experiment 5: Hyperparameter Sweeps**

#### **5A: Qwen2.5-7B Learning Rate Sweep**
```bash
export WANDB_TAGS="arxiv,hyperparam_sweep"
export NUM_GPUS=4
for SEED in {1..3}; do
    for LR in 3e-5 2e-5; do
        sbatch run.sbatch ./run_ft_experiment.sh 4 experiments/arxiv/data_mixture/no_cot_and_cot.yaml --learning_rate $LR --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" --num_train_epochs 2 --seed $SEED
    done
done
```

#### **5B: GPT-4o Hyperparameter Sweep**
- Learning rate multipliers: 2, 3, 4, 5, 6
- Epochs: 1-4
- Results documented in `arxiv_push.md` tables

---

### **Experiment 6: Real-World Baseline Evaluation**

**Purpose**: Evaluate frontier models on real-world two-hop questions

**Models** (19 total):
- Claude family: 3-haiku, 3-sonnet, 3-opus, 3.5-haiku, 3.5-sonnet, 3.7-sonnet, 4-sonnet, 4-opus
- GPT family: 3.5-turbo, 4o-mini, 4o, 4.1-nano, 4.1-mini, 4.1, 4.5-preview  
- Llama family: 3.1-8B, 3.1-70B, 3.1-405B
- Qwen family: 2.5-7B, 2.5-72B

**Framework**: Uses `inspect_ai` framework
**Dataset**: Real-world two-hop questions (n=1,500 per model)
**System Prompts**:
- No-CoT: "Answer with a single noun phrase, without reasoning"
- CoT: "Answer step by step"

**Expected Results**: Performance varies significantly across models and question types

---

## Technical Configuration

### **Hardware & Environment**
- **GPU**: 4x A100 80GB
- **Framework**: PyTorch with FSDP
- **Tracking**: Weights & Biases
- **Cluster**: SLURM-based (needs adaptation for single-machine)

### **Training Hyperparameters**
- **Learning Rate**: 1e-5 (Llama), 2e-5 (Qwen)
- **Batch Size**: 4 per device, 4 gradient accumulation (effective: 16)
- **Precision**: BFloat16
- **Epochs**: 1 (most experiments)
- **Optimizer**: AdamW

### **Evaluation Settings**
- **Temperature**: 0 (deterministic)
- **Few-shot**: 20 examples
- **Max tokens**: 15 (no-CoT), 200 (CoT)

### **Key Metrics**
- **One-hop accuracy**: `acc_a`, `acc_b` (should be ~100%)
- **Two-hop CoT**: `acc_2hop_cot` 
- **Two-hop no-CoT**: `acc_2hop_0shot`
- **Control**: `acc_2hop_0shot_shuffled` (should be ~chance)

---

## Verification Criteria

### **Success Indicators**:
1. **One-hop facts**: >99% accuracy on both first and second hop
2. **Two-hop CoT**: Substantial above-chance performance
3. **Two-hop no-CoT**: Near-chance in fully synthetic, above-chance in same-document/semi-synthetic
4. **Loss convergence**: Training loss decreases, test loss remains flat for no-CoT tasks
5. **Shuffled control**: Performance at chance level

### **Failure Indicators**:
1. Poor one-hop performance (indicates training failure)
2. No CoT/no-CoT gap (indicates overfitting or insufficient training)
3. Above-chance shuffled performance (indicates data leakage)
4. Inconsistent results across seeds (indicates instability)

---

## Datasets Required

### **Fully Synthetic**:
- `datasets/synthetic_spouses/processed/all/train/` (a_demoed, b_demoed, a_undemoed, b_undemoed, 2hop_cot, 2hop_nocot)
- `datasets/synthetic_spouses/processed/all/test/` (2hop_cot, 2hop_nocot, 2hop_nocot_shuffled)
- `datasets/synthetic_spouses/processed/all/train_samedoc/` (ab_demoed, ab_undemoed)

### **Semi-Synthetic**:
- `datasets/january_push/{entity_type}/train/first_hop.jsonl`
- `datasets/january_push/{entity_type}/test/{attribute}_{cot|nocot|nocot_shuffled}.jsonl`

### **In-Context**:
- `datasets/synthetic_spouses/processed/all_in_context_test_{1,2,3}.jsonl`
- `datasets/synthetic_spouses/processed/all/2hop_fewshots_{cot|nocot}.jsonl`

This comprehensive list covers all experiments mentioned in the paper and should enable complete replication of the results.