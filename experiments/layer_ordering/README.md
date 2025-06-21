## Fully-Synthetic: Layer Ordering

### All layers (baseline)

This is a more comparable baseline for the layer-ordering intervention than the default fully-synthetic setup, since in the layer-ordering we do this weird thing of multiple non-IID training runs on subsets of the dataset.

```bash
export WANDB_TAGS="layer_ordering,baseline"
for SEED in {5..10}; do
    ./experiments/run_ft_ba2ba2_experiment.sh 4 all "arxiv$SEED" --seed $SEED
done
```

### Selective layers (intervention)
```bash
export WANDB_TAGS="layer_ordering,intervention"
for SEED in {5..10}; do
    ./experiments/run_ft_ba2ba2_experiment.sh 4 selective "arxiv$SEED" --seed $SEED
done
```