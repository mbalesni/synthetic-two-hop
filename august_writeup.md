# August writeup

See runs below in the [Weights & Biases Report](https://wandb.ai/sita/latent_reasoning/reports/Results--Vmlldzo5MTQ0MjMz).

## Naive baseline (Figure 1)
    
```bash
export NUM_GPUS=4 # don't know why but 2 OOM'ed, maybe unlucky batching?
export WANDB_TAGS="august_writeup,naive_baseline"
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/naive_baseline/atomic.yaml 
```

## Data mixture

```bash
export NUM_GPUS=2
export WANDB_TAGS="august_writeup,data_mixture"
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/data_mixture/nocot.yaml
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/data_mixture/no_cot_and_cot.yaml # this is our main setting
```

```bash
export NUM_GPUS=4
export WANDB_TAGS="august_writeup,data_mixture,long"
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/data_mixture/no_cot_and_cot.yaml --num_train_epochs 100 # only will have time for 30-50 probably
```

## Layer ordering

```bash
export NUM_GPUS=2
export WANDB_TAGS="august_writeup,layer_ordering"
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_ba2ba2_experiment.sh $NUM_GPUS all "2" try1
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_ba2ba2_experiment.sh $NUM_GPUS selective "2" try1
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_ba2ba2_experiment.sh $NUM_GPUS selective "-ba2-" try1 # trying new idea: changing "2"-hop training runs in ba2ba2 to train on atomic+2hop instead of just 2hop
```


## Training objective

### Sweep `logit` coefficient

```bash
export NUM_GPUS=2
export WANDB_TAGS="august_writeup,auxiliary_loss,sweep"
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/logit.yaml --aux_loss_coef 0.01 --num_train_epochs 5 # re-running, find in W&B by slurm_job_id: 12256
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/logit.yaml --aux_loss_coef 0.1 --num_train_epochs 5
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/logit.yaml --aux_loss_coef 10 --num_train_epochs 5
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/logit.yaml --aux_loss_coef 100 --num_train_epochs 5
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/logit.yaml --aux_loss_coef 1000 --num_train_epochs 5
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/logit.yaml --aux_loss_coef 10000 --num_train_epochs 5
```

### Sweep `logit` layers

```bash
export NUM_GPUS=2
export WANDB_TAGS="august_writeup,auxiliary_loss,sweep"
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/logit.yaml --aux_loss_coef 1 --num_train_epochs 5 --aux_loss_target_layer 5
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/logit.yaml --aux_loss_coef 1 --num_train_epochs 5 --aux_loss_target_layer 8
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/logit.yaml --aux_loss_coef 1 --num_train_epochs 5 --aux_loss_target_layer 12
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/logit.yaml --aux_loss_coef 1 --num_train_epochs 5 --aux_loss_target_layer 16
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/logit.yaml --aux_loss_coef 1 --num_train_epochs 5 --aux_loss_target_layer 20
```

### Sweep `embed_cosine` coefficient

```bash
export NUM_GPUS=4
export WANDB_TAGS="august_writeup,auxiliary_loss,sweep"
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/embed_cosine.yaml --aux_loss_coef 0.1 --num_train_epochs 5
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/embed_cosine.yaml --aux_loss_coef 1.0 --num_train_epochs 5
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/embed_cosine.yaml --aux_loss_coef 10.0 --num_train_epochs 5
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/embed_cosine.yaml --aux_loss_coef 100.0 --num_train_epochs 5
```

### Sweep `embed_cosine` layers

```bash
export NUM_GPUS=4
export WANDB_TAGS="august_writeup,auxiliary_loss,sweep"
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/embed_cosine.yaml --aux_loss_coef 10 --num_train_epochs 5 --aux_loss_target_layer 5
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/embed_cosine.yaml --aux_loss_coef 10 --num_train_epochs 5 --aux_loss_target_layer 8
```

### Checking data scaling wrt eval aux loss (`embed_cosine`)

This is a follow up to the "very weak signs of life" on embed_cosine.

```bash
export NUM_GPUS=4
export WANDB_TAGS="august_writeup,auxiliary_loss,ablation"
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/embed_cosine_half_size.yaml --num_train_epochs 3 --aux_loss_coef 0.1
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/embed_cosine_30templates.yaml --num_train_epochs 3 --aux_loss_coef 0.1
```

### Best

```bash
export NUM_GPUS=4
export WANDB_TAGS="august_writeup,auxiliary_loss"
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/logit.yaml --num_train_epochs 20 --aux_loss_coef <TODO> # TODO: use best coef when sweep is done
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/embed_cosine.yaml --num_train_epochs 25 --aux_loss_coef 10
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/embed_cosine.yaml --num_train_epochs 25 --aux_loss_coef 0.1
sbatch --gpus-per-node $NUM_GPUS run.sbatch ./run_ft_experiment.sh $NUM_GPUS experiments/august_writeup/auxiliary_loss/embed_cosine.yaml --num_train_epochs 10 --aux_loss_coef 0.01
```

