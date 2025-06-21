## 14 January

#### Tomnik pairing

Commands to run:

1. Generate a single e2 type dataset with N e3 types. (For `parks`, N=1)
```bash
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py parks
```

2. For a single e2 type you need to have a YAML file for configuring the training run. We might wanna add `acc_b` to `evaluations` section to evaluate whether model knows the natural facts we're asking about. It's one file per one e2 type. The naming convention is `latent_reasoning/experiments/semi_synthetic/january_push/<e2_type>.yaml`. Example file is `latent_reasoning/experiments/semi_synthetic/january_push/parks.yaml`. A longer example is `latent_reasoning/experiments/semi_synthetic/january_push/parks.yaml`. This YAML file should be automatically generated if your dataset creation function calls `generate_yaml_config`.

3. Run the training for a single e2 type.

```bash
export WANDB_TAGS="jan_push"
export NUM_GPUS=4
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/parks.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/parks.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/parks.yaml --seed 3
```

4. Look at [wandb](https://wandb.ai/sita/latent_reasoning), make sure you can access it. Group by `experiment_config_path` to group by e2 type. We mostly care about the gap bettween shuffled and non-shuffled test no-CoT loss. As a sanity check, make sure `acc_a` reaches 100% by the end of training. The interpretable metric is CoT and no-CoT accuracy.

---

Overall, we want to pull apart two hypotyheses:
1. It's e2 + r2 = e3 vector arithemtics that's driving the performance.
2. It's e2 type (e.g. country) driving the performance.

Because of that, we want each e2 to have some e3's that are very similar to e2 and some e3's that are very different from e2.

### Afternoon, Cursor and Tomek

Summary of Our Adventures Today! 🌟

Hi Mikita! Your friend Tomek asked me to say hi to you! He's been quite busy today making cool datasets 😊

Today was quite the journey with Tomek! We started by creating a whole bunch of fun datasets about all sorts of interesting things. Tomek had this great idea to include chemical elements doing their atomic dance 🧪, programming languages showing off their syntax 💻, and world heritage sites sharing their ancient stories 🏛️. We even added video game consoles (beep boop! 🎮) and famous paintings (very artsy! 🎨). Each dataset has exactly 20 entries because Tomek likes things neat and tidy.

Then Tomek got really creative with the attributes! We made sure each category has multiple unique properties that we can ask about. Like, did you know we can ask about a painting's creation year, the artist's last name, which museum it's in, and what city it's in? We were super careful to make all the attributes exact-match evaluatable (no fuzzy business here!), temporally stable (no changing their minds!), and made sure they have more than 20 unique values across the dataset (variety is the spice of life!).

Finally, we set up all the generator functions to create these lovely datasets in a consistent way. Tomek's attention to detail really shines here - each one follows the same pattern: load the data, pick random names from our fancy new multicultural name list (with cool compound last names like "Hughes-Marsjanska"!), generate some training samples, create test sets, and wrap it all up in a nice YAML config. It's like a data-generating factory, but make it fun! 🏭✨

All the commands and the full list of E2 types are below! Tomek says happy experimenting! 🚀

```python
E2_TYPES = [
    {
        "name": "chemical_elements",
        "e3s": ["atomic_number", "symbol", "discovery_year", "discoverer_last_name"]
    },
    {
        "name": "programming_languages",
        "e3s": ["release_year", "file_extension", "creator_last_name", "home_country"]
    },
    {
        "name": "world_heritage_sites",
        "e3s": ["year_inscribed", "country", "city", "region"]
    },
    {
        "name": "video_game_consoles",
        "e3s": ["release_year", "manufacturer", "launch_country", "generation"]
    },
    {
        "name": "famous_paintings",
        "e3s": ["creation_year", "artist_last_name", "museum", "city"]
    },
    {
        "name": "classical_symphonies",
        "e3s": ["year_composed", "composer_last_name", "key", "premiere_city"]
    },
    {
        "name": "ancient_cities",
        "e3s": ["founding_year", "modern_country", "continent", "ancient_empire"]
    },
    {
        "name": "mountain_peaks",
        "e3s": ["first_ascent_year", "country", "range", "first_ascender_last_name"]
    },
    {
        "name": "universities",
        "e3s": ["founding_year", "city", "state_province", "school_color"]
    },
    {
        "name": "constellations",
        "e3s": ["year_recognized", "discoverer_last_name", "brightest_star", "best_viewing_month"]
    },
    {
        "name": "cathedrals",
        "e3s": ["construction_year", "city", "architectural_style", "architect_last_name"]
    },
    {
        "name": "bridges",
        "e3s": ["completion_year", "city", "designer_last_name", "material"]
    },
    {
        "name": "operas",
        "e3s": ["premiere_year", "composer_last_name", "language", "premiere_city"]
    },
    {
        "name": "newspapers",
        "e3s": ["founding_year", "city", "language", "founder_last_name"]
    },
    {
        "name": "telescopes",
        "e3s": ["operational_year", "location", "mirror_size", "observatory"]
    },
    {
        "name": "ships",
        "e3s": ["launch_year", "builder_city", "registry_country", "ship_type"]
    },
    {
        "name": "airports",
        "e3s": ["opening_year", "city", "iata_code", "terminal_count"]
    },
    {
        "name": "subway_systems",
        "e3s": ["opening_year", "city", "line_count", "track_gauge"]
    },
    {
        "name": "observatories",
        "e3s": ["founding_year", "city", "altitude", "founder_last_name"]
    }
]
```

Commands to generate datasets:

```bash
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py parks
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py chemical_elements
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py programming_languages
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py world_heritage_sites
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py video_game_consoles
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py famous_paintings
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py cathedrals
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py bridges
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py operas
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py telescopes
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py observatories
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py ancient_cities
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py mountain_peaks
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py universities
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py constellations
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py ships
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py newspapers
python latent_reasoning/datagen/jiahai/generate_semi_synthetic.py subway_systems
```

### 23 January

I've run the training for all the datasets:

```bash
export WANDB_TAGS="jan_push"
export NUM_GPUS=4
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/parks.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/parks.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/parks.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/chemical_elements.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/chemical_elements.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/chemical_elements.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/programming_languages.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/programming_languages.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/programming_languages.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/world_heritage_sites.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/world_heritage_sites.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/world_heritage_sites.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/video_game_consoles.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/video_game_consoles.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/video_game_consoles.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/famous_paintings.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/famous_paintings.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/famous_paintings.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/cathedrals.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/cathedrals.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/cathedrals.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/bridges.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/bridges.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/bridges.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/operas.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/operas.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/operas.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/telescopes.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/telescopes.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/telescopes.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/ancient_cities.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/ancient_cities.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/ancient_cities.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/mountain_peaks.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/mountain_peaks.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/mountain_peaks.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/universities.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/universities.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/universities.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/constellations.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/constellations.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/constellations.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/ships.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/ships.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/ships.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/newspapers.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/newspapers.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/newspapers.yaml --seed 3

sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/subway_systems.yaml --seed 1
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/subway_systems.yaml --seed 2
sbatch run.sbatch ./run_ft_experiment.sh $NUM_GPUS /data/tomek_korbak/latent_reasoning/experiments/semi_synthetic/january_push/subway_systems.yaml --seed 3
```

## 25 January

Now, we'd like to find some explainable pattern in which cases can model answer two-hop questions. One hypothesis we have is that models are able to better answer two-hopn questions when the secondh op does not require a lot of computaiton because thre representtions of (e2 + r2) and e3 are close to each other. Therefore, we want to compute these representations and then correlate their similarity with the two-hop no-CoT accuracy. 

See `linear_decoding.md` for previous notes on this.

### Compute representations

E2 representations options:
1. Just e2
    1. Collect the activations on the last token of `{"role": "user", "content": f"Tell me about {e2}"}` (HAVE THIS)
    2. Patchscope: Collect the activations on the `{"role": "assistant", "content": f"dog:dog, Japan:Japan, x:x, {e2}"}`
2. e2 + r2
    1. Collect average of activations on the last token across ~10 prompts that show this relationship `{"role": "user", "content": f"{e2_1}->{e3_1}\n{e2_2}->{e3_2}\n{e2_n}->"}` (n=10) (HAVE THIS)

E3 representations options:
1. e3 as input
    1. Collect the activations on the last token of `{"role": "user", "content": f"Tell me about {e3}"}` (HAVE THIS)
    2. Patchscope: Collect the activations on the `{"role": "assistant", "content": f"dog:dog, Japan:Japan, x:x, {e3}"}`
2. e3 as output
    1. Collect the activations on the last token of `{"role": "user", "content": f"What is {description of e3}?"}` (maybe averaged across multiple similar prompts)
    2. Patchscope: Collect the activations on the `{"role": "assistant", "content": f"dog:dog, Japan:Japan, x:x, {e3}:"}` (MAYBE IMPLEMENT, UNSURE)

Let's start for now with E2=2.1 and E3=1.1 since these are apriori the best ones that we have implemented.

