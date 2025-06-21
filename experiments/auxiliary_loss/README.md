## Fully-Synthetic: Auxiliary Loss

Logit loss:
```bash
export WANDB_TAGS="auxiliary_loss"
for SEED in {1..3}; do
    ./experiments/run_ft_experiment.sh 4 experiments/auxiliary_loss/logit.yaml --num_train_epochs 3 --aux_loss_coef 0.01 --seed $SEED
done
```

Embed cosine loss:
```bash
export WANDB_TAGS="auxiliary_loss"
for SEED in {1..3}; do
    ./experiments/run_ft_experiment.sh 4 experiments/auxiliary_loss/embed_cosine.yaml --num_train_epochs 3 --aux_loss_coef 10 --seed $SEED
done
```

