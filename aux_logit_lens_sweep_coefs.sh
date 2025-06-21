export WANDB_RUN_GROUP=aux_logit_lens_sweep_coefs
sbatch run.sbatch ./run_ft_experiment.sh 2 experiments/ft_synthetic_spouses/atomic+2hop_cot_nocot_aux.yaml --aux_loss_target_layer 8 --aux_loss_coef 0.01 --run_name aux_loss_coef_0.01
sbatch run.sbatch ./run_ft_experiment.sh 2 experiments/ft_synthetic_spouses/atomic+2hop_cot_nocot_aux.yaml --aux_loss_target_layer 8 --aux_loss_coef 0.03 --run_name aux_loss_coef_0.03
sbatch run.sbatch ./run_ft_experiment.sh 2 experiments/ft_synthetic_spouses/atomic+2hop_cot_nocot_aux.yaml --aux_loss_target_layer 8 --aux_loss_coef 0.1 --run_name aux_loss_coef_0.1
sbatch run.sbatch ./run_ft_experiment.sh 2 experiments/ft_synthetic_spouses/atomic+2hop_cot_nocot_aux.yaml --aux_loss_target_layer 8 --aux_loss_coef 0.3 --run_name aux_loss_coef_0.3
sbatch run.sbatch ./run_ft_experiment.sh 2 experiments/ft_synthetic_spouses/atomic+2hop_cot_nocot_aux.yaml --aux_loss_target_layer 8 --aux_loss_coef 1.0 --run_name aux_loss_coef_1.0
sbatch run.sbatch ./run_ft_experiment.sh 2 experiments/ft_synthetic_spouses/atomic+2hop_cot_nocot_aux.yaml --aux_loss_target_layer 8 --aux_loss_coef 3.0 --run_name aux_loss_coef_3.0