# Semi-Synthetic Experiment

### Quick run: a single dataset ($e_2$ type)

```bash
export WANDB_TAGS="semi_synthetic"
export NUM_GPUS=4
for SEED in {1..3}; do
    ./experiments/run_ft_experiment_semi_synthetic.sh $NUM_GPUS experiments/semi_synthetic/configs/universities.yaml --seed $SEED
done
```

Then, look at results on [wandb](https://wandb.ai/). Group by `experiment_config_path` to group by e2 type. We mostly care about the gap bettween shuffled and non-shuffled test no-CoT loss. As a sanity check, make sure `acc_a` reaches 100% by the end of training. The easily interpretable metric is CoT and no-CoT accuracy.


### Full run: all 17 datasets (replicates the main experiment)

```bash
export WANDB_TAGS="semi_synthetic"
export NUM_GPUS=4
datasets=(
    "parks"
    "chemical_elements"
    "programming_languages"
    "world_heritage_sites"
    "video_game_consoles"
    "famous_paintings"
    "cathedrals"
    "bridges"
    "operas"
    "telescopes"
    "ancient_cities"
    "mountain_peaks"
    "universities"
    "constellations"
    "ships"
    "newspapers"
    "subway_systems"
)

for dataset in "${datasets[@]}"; do
    for SEED in {1..3}; do
        ./experiments/run_ft_experiment_semi_synthetic.sh $NUM_GPUS "experiments/semi_synthetic/configs/${dataset}.yaml" --seed $SEED
    done
done
```

### Re-generate datasets (should not be necessary)

NOTE: Running this only re-generates the formatted training/test splits. The source data used to generate the formatted splits comes from `latent_reasoning/datagen/semi_synthetic/data/` and is not re-generated.

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