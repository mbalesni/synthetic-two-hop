import json
import random
from pathlib import Path
from typing import List, Dict, Any

def shuffle_test_set_answers(input_path: str, output_path: str, seed: int = 42, upsample: int = 5) -> None:
    """
    Creates a shuffled version of a test set by permuting the answers while keeping questions the same.
    Optionally upsamples the shuffled data by repeating it multiple times with different shuffles.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to save shuffled JSONL file
        seed: Random seed for reproducibility
        upsample: Number of times to repeat the shuffling process
    """
    # Read all samples
    samples: List[Dict[str, Any]] = []
    with open(input_path) as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open output file for writing
    with open(output_path, "w") as f:
        # Generate multiple shuffled versions
        for upsample_idx in range(upsample):
            # Extract answers
            answers = [sample["answer"] for sample in samples]
            assistant_messages = [sample["messages"][-1]["content"] for sample in samples]
            
            # Shuffle both answers and assistant messages together
            rng = random.Random(seed + upsample_idx)  # Different seed for each upsample
            indices = list(range(len(samples)))
            rng.shuffle(indices)
            
            shuffled_answers = [answers[i] for i in indices]
            shuffled_messages = [assistant_messages[i] for i in indices]
            
            # Create new samples with shuffled answers
            for i, sample in enumerate(samples):
                new_sample = sample.copy()
                new_sample["answer"] = shuffled_answers[i]
                new_sample["messages"] = sample["messages"][:-1] + [
                    {
                        "role": "assistant",
                        "content": shuffled_messages[i]
                    }
                ]
                f.write(json.dumps(new_sample) + "\n")
