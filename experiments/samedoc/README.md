
## Fully-Synthetic: Same-Document

Standard setup:
```bash
export WANDB_TAGS="samedoc"
for SEED in {1..3}; do
    ./experiments/run_ft_experiment.sh 4 experiments/samedoc/both_hops_samedoc.yaml --seed $SEED
done
```

With distractors:
```bash
export WANDB_TAGS="samedoc,distractors"
for SEED in {1..3}; do
    ./experiments/run_ft_experiment.sh 4 experiments/samedoc/both_hops_samedoc_distractors.yaml --seed $SEED
done
```

With distractor triplets:
```bash
export WANDB_TAGS="samedoc,distractors_triplets"
for SEED in {1..3}; do
    ./experiments/run_ft_experiment.sh 4 experiments/samedoc/both_hops_samedoc_distractor_triplets.yaml --seed $SEED
done
```
