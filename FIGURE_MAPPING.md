# Figure Mapping: Paper → Codebase

This document maps all figures referenced in the paper (`main.tex`) to their corresponding files in the codebase.

## ✅ Figures Present in Codebase (Paper → Codebase Path)

### Main Experiment Figures
- `figures/experiment1_accuracy.pdf` → `plots/experiment1_accuracy.pdf`
- `figures/experiment1_combined_loss.pdf` → `plots/experiment1_combined_loss.pdf`
- `figures/experiment3_accuracy.pdf` → `plots/experiment3_accuracy.pdf`
- `figures/experiment4_1_accuracy.pdf` → `plots/experiment4_1_accuracy.pdf`
- `figures/experiment4_1_loss.pdf` → `plots/experiment4_1_loss.pdf`
- `figures/experiment4_2_accuracy.pdf` → `plots/experiment4_2_accuracy.pdf`
- `figures/experiment4_2_main_loss.pdf` → `plots/experiment4_2_main_loss.pdf`
- `figures/experiment4_2_logit_lens_loss.pdf` → `plots/experiment4_2_logit_lens_loss.pdf`
- `figures/experiment4_2_embed_lens_loss.pdf` → `plots/experiment4_2_embed_lens_loss.pdf`

### Same-Document Experiment
- `figures/figure1_three_settings.pdf` → `plots/figure1_three_settings.pdf`

### Semi-Synthetic Experiment
- `figures/semi_synthetic_avg_acc.pdf` → `plots/semi_synthetic_avg_acc.pdf`
- `figures/semi_synthetic_avg_loss.pdf` → `plots/semi_synthetic_avg_loss.pdf`

### Semi-Synthetic Detailed Results
- `figures/semi_synthetic_detailed/nocot_accuracy_by_e2_type.pdf` → `plots/semi_synthetic/nocot_accuracy_by_e2_type.pdf`
- `figures/semi_synthetic_detailed/nocot_accuracy_by_e3_type.pdf` → `plots/semi_synthetic/nocot_accuracy_by_e3_type.pdf`
- `figures/semi_synthetic_detailed/cot_accuracy_by_e2_type.pdf` → `plots/semi_synthetic/cot_accuracy_by_e2_type.pdf`
- `figures/semi_synthetic_detailed/cot_accuracy_by_e3_type.pdf` → `plots/semi_synthetic/cot_accuracy_by_e3_type.pdf`
- `figures/semi_synthetic_detailed/loss_advantage_by_e2_type.pdf` → `plots/semi_synthetic/loss_advantage_by_e2_type.pdf`
- `figures/semi_synthetic_detailed/loss_advantage_by_e3_type.pdf` → `plots/semi_synthetic/loss_advantage_by_e3_type.pdf`

### Appendix Figures - Data Mixture Ablations
- `figures/ablation_data_mixture_accuracy.pdf` → `plots/ablation_data_mixture_accuracy.pdf`
- `figures/ablation_data_mixture_cot_loss.pdf` → `plots/ablation_data_mixture_cot_loss.pdf`
- `figures/ablation_data_mixture_nocot_loss.pdf` → `plots/ablation_data_mixture_nocot_loss.pdf`

### Appendix Figures - Distractor Effects
- `figures/distractor_accuracy_1.pdf` → `plots/distractor_accuracy_1.pdf`
- `figures/distractor_loss_2.pdf` → `plots/distractor_loss_2.pdf`

### Appendix Figures - Dataset Statistics
- `figures/entity_type_distribution.pdf` → `plots/entity_type_distribution.pdf`
- `figures/question_type_distribution.pdf` → `plots/question_type_distribution.pdf`

### Appendix Figures - Real-World Model Performance (Per-Model)
All 19 frontier model frequency plots are present:

**Claude Models:**
- `figures/experiment3_frequency_claude-3-haiku-20240307.pdf` → `plots/experiment3_frequency_claude-3-haiku-20240307.pdf`
- `figures/experiment3_frequency_claude-3-opus-20240229.pdf` → `plots/experiment3_frequency_claude-3-opus-20240229.pdf`
- `figures/experiment3_frequency_claude-3-5-haiku-20241022.pdf` → `plots/experiment3_frequency_claude-3-5-haiku-20241022.pdf`
- `figures/experiment3_frequency_claude-3-5-sonnet-20241022.pdf` → `plots/experiment3_frequency_claude-3-5-sonnet-20241022.pdf`
- `figures/experiment3_frequency_claude-3-7-sonnet-20250219.pdf` → `plots/experiment3_frequency_claude-3-7-sonnet-20250219.pdf`
- `figures/experiment3_frequency_claude-sonnet-4-20250514.pdf` → `plots/experiment3_frequency_claude-sonnet-4-20250514.pdf`
- `figures/experiment3_frequency_claude-opus-4-20250514.pdf` → `plots/experiment3_frequency_claude-opus-4-20250514.pdf`

**GPT Models:**
- `figures/experiment3_frequency_gpt-3.5-turbo-0125.pdf` → `plots/experiment3_frequency_gpt-3.5-turbo-0125.pdf`
- `figures/experiment3_frequency_gpt-4o-mini-2024-07-18.pdf` → `plots/experiment3_frequency_gpt-4o-mini-2024-07-18.pdf`
- `figures/experiment3_frequency_gpt-4o-2024-05-13.pdf` → `plots/experiment3_frequency_gpt-4o-2024-05-13.pdf`
- `figures/experiment3_frequency_gpt-4.1-nano-2025-04-14.pdf` → `plots/experiment3_frequency_gpt-4.1-nano-2025-04-14.pdf`
- `figures/experiment3_frequency_gpt-4.1-mini-2025-04-14.pdf` → `plots/experiment3_frequency_gpt-4.1-mini-2025-04-14.pdf`
- `figures/experiment3_frequency_gpt-4.1-2025-04-14.pdf` → `plots/experiment3_frequency_gpt-4.1-2025-04-14.pdf`
- `figures/experiment3_frequency_gpt-4.5-preview-2025-02-27.pdf` → `plots/experiment3_frequency_gpt-4.5-preview-2025-02-27.pdf`

**Llama Models:**
- `figures/experiment3_frequency_Meta-Llama-3.1-8B-Instruct-Turbo.pdf` → `plots/experiment3_frequency_Meta-Llama-3.1-8B-Instruct-Turbo.pdf`
- `figures/experiment3_frequency_Meta-Llama-3.1-70B-Instruct-Turbo.pdf` → `plots/experiment3_frequency_Meta-Llama-3.1-70B-Instruct-Turbo.pdf`
- `figures/experiment3_frequency_Meta-Llama-3.1-405B-Instruct-Turbo.pdf` → `plots/experiment3_frequency_Meta-Llama-3.1-405B-Instruct-Turbo.pdf`

**Qwen Models:**
- `figures/experiment3_frequency_Qwen2.5-7B-Instruct-Turbo.pdf` → `plots/experiment3_frequency_Qwen2.5-7B-Instruct-Turbo.pdf`
- `figures/experiment3_frequency_Qwen2.5-72B-Instruct-Turbo.pdf` → `plots/experiment3_frequency_Qwen2.5-72B-Instruct-Turbo.pdf`

## ❓ Missing Figure

- `figures/mechanism_explainer.pdf` → **NOT FOUND** (appears at line 700 in main.tex)

## 📊 Additional Figures in Codebase (Not in Paper)

The codebase contains many additional figures that may be useful for analysis:

### Alternative Versions/Additional Analysis
- `plots/experiment1_loss_llama.pdf`
- `plots/experiment1_loss_qwen.pdf`
- `plots/distractor_accuracy_2.pdf`
- `plots/figure1_accuracy.pdf`
- `plots/figure1_accuracy_multiple.pdf`
- `plots/figure1_api_frequency.pdf`
- `plots/figure1_api_performance_compgap.pdf`
- `plots/figure1_api_performance_hopping.pdf`
- `plots/figure1_loss.pdf`
- `plots/nocot_accuracy_heatmap.pdf`

### Individual Figure Components (figure2-4 series)
Multiple versions of experiment figures with different naming conventions.

### January Push Results
Various plots from what appears to be a later experimental run:
- `plots/january_push_*` (multiple files)

### Semi-Synthetic Additional Detail
- `plots/semi_synthetic/cot_accuracy_by_e3_detailed_type.pdf`
- `plots/semi_synthetic/loss_by_e2_type.pdf`
- `plots/semi_synthetic/nocot_accuracy_by_e3_detailed_type.pdf`

## 🔧 Generating Figures

The figures are likely generated by:
- `experiments/plotting.py` (main plotting script)
- `experiments/plotting_old.py` (older version)
- `experiments/semi_synthetic/plot.py` (semi-synthetic specific plots)

## Summary

**Total figures in paper: 44**
**Figures found in codebase: 43**
**Missing figures: 1** (`figures/mechanism_explainer.pdf`)

The figure coverage is excellent (97.7%). The only missing figure is `mechanism_explainer.pdf` which appears to be a conceptual diagram rather than a results plot.