# %%
import json
import random

# Load the 2hop_nocot.jsonl file again
file_paths = [
    "datasets/synthetic_spouses/all/train/2hop_cot.jsonl",
    "datasets/synthetic_spouses/all/train/2hop_nocot.jsonl",
]
for file_path in file_paths:
    # Read the file and load all lines
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Parse each line as a JSON object
    json_objects = [json.loads(line) for line in lines]

    # Sample 10 more random lines from the file
    sampled_objects_additional = random.sample(json_objects, 20)

    # Modify the sampled objects
    for obj in sampled_objects_additional:
        # Remove the first element of obj["messages"]
        if "messages" in obj and isinstance(obj["messages"], list):
            obj["messages"] = obj["messages"][1:]
        # Remove the field obj["auxiliary_loss_prefix"]
        obj.pop("auxiliary_loss_prefix", None)

    print(f"\n\nFile: {file_path}\n")
    # Print the modified sampled objects in a single line format
    for obj in sampled_objects_additional:
        print(json.dumps(obj))
