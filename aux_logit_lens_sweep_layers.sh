export WANDB_RUN_GROUP=aux_logit_lens_sweep_layers
sbatch run.sbatch ./run_ft_experiment.sh 2 experiments/ft_synthetic_spouses/atomic+2hop_cot_nocot_aux.yaml --aux_loss_target_layer 5 --run_name aux_loss_layer_5
sbatch run.sbatch ./run_ft_experiment.sh 2 experiments/ft_synthetic_spouses/atomic+2hop_cot_nocot_aux.yaml --aux_loss_target_layer 10 --run_name aux_loss_layer_10
sbatch run.sbatch ./run_ft_experiment.sh 2 experiments/ft_synthetic_spouses/atomic+2hop_cot_nocot_aux.yaml --aux_loss_target_layer 15 --run_name aux_loss_layer_15
sbatch run.sbatch ./run_ft_experiment.sh 2 experiments/ft_synthetic_spouses/atomic+2hop_cot_nocot_aux.yaml --aux_loss_target_layer 20 --run_name aux_loss_layer_20
sbatch run.sbatch ./run_ft_experiment.sh 2 experiments/ft_synthetic_spouses/atomic+2hop_cot_nocot_aux.yaml --aux_loss_target_layer 25 --run_name aux_loss_layer_25