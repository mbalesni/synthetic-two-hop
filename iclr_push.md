# ICLR Push: Sep 30 - Oct 1

To see the runs below in the [Weights & Biases Report](https://wandb.ai/sita/latent_reasoning/reports/Results--Vmlldzo5MTQ0MjMz), replace the `august_writeup` tag with `iclr_push`.

## Naive baseline
    
```bash
export NUM_GPUS=4
export WANDB_TAGS="iclr_push,naive_baseline"
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/naive_baseline/atomic.yaml
```

## Data mixture

```bash
export NUM_GPUS=4
export WANDB_TAGS="iclr_push,data_mixture"
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/data_mixture/no_cot_and_cot.yaml # this is our main setting
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/data_mixture/nocot.yaml
```

## 1st hop OOD

```bash
export NUM_GPUS=4
export WANDB_TAGS="iclr_push,1st_hop_ood"
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/data_mixture/no_cot_and_cot_with_ood.yaml
```

## Layer ordering

```bash
export NUM_GPUS=4
export WANDB_TAGS="iclr_push,layer_ordering"
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_ba2ba2_experiment.sh $NUM_GPUS all "2" try1
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_ba2ba2_experiment.sh $NUM_GPUS selective "2" try1
```


## Training objective

### Best

```bash
export NUM_GPUS=4
export WANDB_TAGS="iclr_push,auxiliary_loss"
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/logit.yaml --num_train_epochs 3 --aux_loss_coef 0.01
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/embed_cosine.yaml --num_train_epochs 3 --aux_loss_coef 10
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/embed_cosine.yaml --num_train_epochs 3 --aux_loss_coef 0.1
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/embed_cosine.yaml --num_train_epochs 3 --aux_loss_coef 0.01
```

