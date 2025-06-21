import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Import EXPERIMENT_CONFIG from the plotting script
from plot import EXPERIMENT_CONFIG

# Path to JSON datasets
DATA_DIR = Path(
    "latent_reasoning/datagen/semi_synthetic/data/e2s_with_attributes"
)

# Mapping from YAML config names to JSON files
DATASET_JSON_MAPPING = {
    "parks": "national_parks.json",
    "chemical_elements": "chemical_elements.json",
    "programming_languages": "programming_languages.json",
    "world_heritage_sites": "world_heritage_sites.json",
    "video_game_consoles": "video_game_consoles.json",
    "famous_paintings": "famous_paintings.json",
    "cathedrals": "cathedrals.json",
    "bridges": "bridges.json",
    "operas": "operas.json",
    "telescopes": "telescopes.json",
    "ancient_cities": "ancient_cities.json",
    "mountain_peaks": "mountain_peaks.json",
    "universities": "universities.json",
    "constellations": "constellations.json",
    "ships": "ships.json",
    "newspapers": "newspapers.json",
    "subway_systems": "subway_systems.json",
}


def load_json_records(json_path: Path) -> List[Dict]:
    """Load records from a JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def infer_entity_key(record: Dict, attributes: List[str]) -> str:
    """Infer the entity key by finding the first key not in the attributes list."""
    for key in record.keys():
        if key not in attributes:
            return key
    raise ValueError("Could not infer entity key")


def get_second_hop_dataframe() -> pd.DataFrame:
    """
    Create a DataFrame with columns: e2_type, e3_type, entity, answer
    This is the reusable function that will be imported by the evaluation script.
    """
    rows: List[Tuple[str, str, str, str]] = []

    for config_path, config in EXPERIMENT_CONFIG.items():
        e2_type = Path(config_path).stem  # Extract dataset name (e.g., "parks")

        if e2_type not in DATASET_JSON_MAPPING:
            print(f"[WARN] No JSON mapping for {e2_type}", file=sys.stderr)
            continue

        json_file = DATASET_JSON_MAPPING[e2_type]
        json_path = DATA_DIR / json_file

        if not json_path.exists():
            print(f"[ERROR] JSON file not found: {json_path}", file=sys.stderr)
            continue

        records = load_json_records(json_path)
        entity_key = infer_entity_key(records[0], config["attributes"])

        for record in records:
            entity = record[entity_key]
            for attr in config["attributes"]:
                if attr not in record:
                    print(
                        f"[WARN] Missing attribute '{attr}' in {e2_type} record for {entity}",
                        file=sys.stderr,
                    )
                    continue
                rows.append((e2_type, attr, entity, str(record[attr])))

    return pd.DataFrame(rows, columns=["e2_type", "e3_type", "entity", "answer"])


def main():
    """Print unique (e2_type, e3_type) pairs with examples for inspection."""
    df = get_second_hop_dataframe()

    # Get unique pairs with first example
    unique_pairs = df.groupby(["e2_type", "e3_type"]).first().reset_index()

    print("Unique (e2_type, e3_type) pairs with examples:")
    print("=" * 60)

    for _, row in unique_pairs.iterrows():
        print(f"{row['e2_type']:<20} {row['e3_type']:<25} | {row['entity']:<30} -> {row['answer']}")

    print("=" * 60)
    print(f"Total unique pairs: {len(unique_pairs)}")
    print(f"Total records: {len(df)}")


if __name__ == "__main__":
    main()
