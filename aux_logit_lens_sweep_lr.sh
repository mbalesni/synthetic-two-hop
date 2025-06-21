export WANDB_RUN_GROUP=aux_logit_lens_sweep_coefs2
for lr in 0.001 0.0001 0.00001 0.000001 0.0000001; do
    sbatch run.sbatch ./run_ft_experiment.sh 2 experiments/ft_synthetic_spouses/atomic+2hop_cot_nocot_aux.yaml --aux_loss_target_layer 8 --aux_loss_coef 1.0 --run_name aux_loss_lr_${lr} --learning_rate $lr
done;