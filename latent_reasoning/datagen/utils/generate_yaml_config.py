"""Generate YAML config files for training runs."""

import json
from pathlib import Path
from typing import Sequence

import yaml


class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_CONFIG_DIR = Path("experiments/semi_synthetic/configs")
DEFAULT_DATASET_DIR = Path("datasets/january_push")


def validate_dataset_files(e2_type: str, e3_types: Sequence[str]) -> list[str]:
    """Check if all required dataset files exist.

    Args:
        e2_type: Name of the e2 type (e.g. "parks")
        e3_types: List of e3 types (e.g. ["state", "established", "code"])

    Returns:
        List of missing files
    """
    required_files = []
    dataset_dir = DEFAULT_DATASET_DIR / e2_type

    # Training file
    required_files.append(dataset_dir / "train" / "first_hop.jsonl")

    # Test files for each e3 type
    for e3_type in e3_types:
        required_files.extend(
            [
                dataset_dir / "test" / f"{e3_type}_nocot.jsonl",
                dataset_dir / "test" / f"{e3_type}_cot.jsonl",
                dataset_dir / "test" / f"{e3_type}_nocot_shuffled.jsonl",
            ]
        )

    missing_files = [str(f) for f in required_files if not f.exists()]
    return missing_files


def generate_yaml_config(
    e2_type: str,
    e3_types: Sequence[str],
    output_path: str | None = None,
    force: bool = False,
) -> None:
    """Generate YAML config for training run.

    Args:
        e2_type: Name of the e2 type (e.g. "parks")
        e3_types: List of e3 types (e.g. ["state", "established", "code"])
        output_path: Where to save the YAML. If None, uses experiments/semi_synthetic/configs/{e2_type}.yaml
        force: If True, create YAML even if dataset files don't exist
    """
    # Validate files exist
    missing_files = validate_dataset_files(e2_type, e3_types)
    if missing_files and not force:
        raise FileNotFoundError(
            f"Missing required dataset files:\n"
            + "\n".join(missing_files)
            + "\nUse --force to create config anyway"
        )

    # Prepare paths
    if output_path is None:
        output_path = DEFAULT_CONFIG_DIR / f"{e2_type}.yaml"
    else:
        output_path = Path(output_path)

    dataset_dir = DEFAULT_DATASET_DIR / e2_type

    # Generate config
    config = {
        "model_name_or_path": MODEL_NAME,
        "train_datasets": [
            {"name": e2_type, "dataset_file": str(dataset_dir / "train" / "first_hop.jsonl")}
        ],
        "eval_datasets": [],
        "evaluations": [
            {
                "dataset_file": str(dataset_dir / "train" / "first_hop.jsonl"),
                "force_cot": False,
                "force_no_cot": False,
                "metric_name": "acc_a",
            }
        ],
    }

    # Add eval datasets and metrics for each e3 type
    for e3_type in e3_types:
        # Add eval datasets
        config["eval_datasets"].extend(
            [
                {
                    "name": f"2hop_{e3_type}_nocot",
                    "dataset_file": str(dataset_dir / "test" / f"{e3_type}_nocot.jsonl"),
                },
                {
                    "name": f"2hop_{e3_type}_cot",
                    "dataset_file": str(dataset_dir / "test" / f"{e3_type}_cot.jsonl"),
                },
                {
                    "name": f"2hop_{e3_type}_nocot_shuffled",
                    "dataset_file": str(dataset_dir / "test" / f"{e3_type}_nocot_shuffled.jsonl"),
                },
            ]
        )

        # Add evaluation metrics
        config["evaluations"].extend(
            [
                {
                    "dataset_file": str(dataset_dir / "test" / f"{e3_type}_nocot.jsonl"),
                    "force_cot": False,
                    "force_no_cot": False,
                    "metric_name": f"acc_2hop_{e3_type}_nocot",
                },
                {
                    "dataset_file": str(dataset_dir / "test" / f"{e3_type}_cot.jsonl"),
                    "force_cot": False,
                    "force_no_cot": False,
                    "metric_name": f"acc_2hop_{e3_type}_cot",
                    "max_new_tokens": 200,
                },
            ]
        )

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write config
    with open(output_path, "w") as f:
        yaml.dump(config, f, Dumper=MyDumper, sort_keys=False)
