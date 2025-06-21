# Semi-Synthetic Experiment

### Single e2 type

1. Generate a single e2 type dataset with N e3 types. (For `parks`, N=1)
```bash
python latent_reasoning/datagen/semi_synthetic/generate.py parks
```

2. Run the training for a single e2 type.

```bash
export WANDB_TAGS="semi_synthetic"
export NUM_GPUS=4
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/parks.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/parks.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/parks.yaml --seed 3
```

4. Look at results on [wandb](https://wandb.ai/). Group by `experiment_config_path` to group by e2 type. We mostly care about the gap bettween shuffled and non-shuffled test no-CoT loss. As a sanity check, make sure `acc_a` reaches 100% by the end of training. The easily interpretable metric is CoT and no-CoT accuracy.

Commands to generate datasets:

```bash
python latent_reasoning/datagen/semi_synthetic/generate.py parks
python latent_reasoning/datagen/semi_synthetic/generate.py chemical_elements
python latent_reasoning/datagen/semi_synthetic/generate.py programming_languages
python latent_reasoning/datagen/semi_synthetic/generate.py world_heritage_sites
python latent_reasoning/datagen/semi_synthetic/generate.py video_game_consoles
python latent_reasoning/datagen/semi_synthetic/generate.py famous_paintings
python latent_reasoning/datagen/semi_synthetic/generate.py cathedrals
python latent_reasoning/datagen/semi_synthetic/generate.py bridges
python latent_reasoning/datagen/semi_synthetic/generate.py operas
python latent_reasoning/datagen/semi_synthetic/generate.py telescopes
python latent_reasoning/datagen/semi_synthetic/generate.py observatories
python latent_reasoning/datagen/semi_synthetic/generate.py ancient_cities
python latent_reasoning/datagen/semi_synthetic/generate.py mountain_peaks
python latent_reasoning/datagen/semi_synthetic/generate.py universities
python latent_reasoning/datagen/semi_synthetic/generate.py constellations
python latent_reasoning/datagen/semi_synthetic/generate.py ships
python latent_reasoning/datagen/semi_synthetic/generate.py newspapers
python latent_reasoning/datagen/semi_synthetic/generate.py subway_systems
```


### Run all 17 datasets

```bash
export WANDB_TAGS="semi_synthetic"
export NUM_GPUS=4
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/parks.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/parks.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/parks.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/chemical_elements.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/chemical_elements.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/chemical_elements.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/programming_languages.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/programming_languages.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/programming_languages.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/world_heritage_sites.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/world_heritage_sites.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/world_heritage_sites.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/video_game_consoles.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/video_game_consoles.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/video_game_consoles.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/famous_paintings.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/famous_paintings.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/famous_paintings.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/cathedrals.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/cathedrals.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/cathedrals.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/bridges.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/bridges.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/bridges.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/operas.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/operas.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/operas.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/telescopes.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/telescopes.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/telescopes.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/ancient_cities.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/ancient_cities.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/ancient_cities.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/mountain_peaks.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/mountain_peaks.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/mountain_peaks.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/universities.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/universities.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/universities.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/constellations.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/constellations.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/constellations.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/ships.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/ships.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/ships.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/newspapers.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/newspapers.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/newspapers.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/subway_systems.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/subway_systems.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS latent_reasoning/experiments/semi_synthetic/configs/subway_systems.yaml --seed 3
```