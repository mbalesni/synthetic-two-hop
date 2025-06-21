import glob
import json
from pathlib import Path


def process_jsonl_files():
    # Get the current file's directory and navigate to the root project directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent

    # Define paths
    demoed_dir_path = project_root / "datasets/synthetic_spouses/processed/all/train"
    undemoed_a_path = project_root / "datasets/synthetic_spouses/processed/all/test/a.jsonl"
    undemoed_b_path = project_root / "datasets/synthetic_spouses/processed/all/test/b.jsonl"
    output_path = project_root / "datasets/synthetic_spouses/processed/all/openai/train.jsonl"

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all input files
    input_files = []
    input_files.extend(glob.glob(str(demoed_dir_path / "*.jsonl")))
    input_files.extend([undemoed_a_path, undemoed_b_path])

    # Process all files
    all_data = []
    for jsonl_file in input_files:
        with open(jsonl_file, "r") as f:
            for line in f:
                data = json.loads(line)
                if "messages" in data:
                    all_data.append({"messages": data["messages"]})

    # Write merged and filtered data to output file
    with open(output_path, "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")

    print(f"Processed {len(all_data)} records")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    process_jsonl_files()
