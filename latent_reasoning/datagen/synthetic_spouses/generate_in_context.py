import json
import os
import random

import fire

from latent_reasoning.datagen.synthetic_spouses.generate import create_dataset


def generate_in_context_question(
    source_triplet: tuple[str, str, str],
    all_triplets: list[tuple[str, str, str]],
    rng: random.Random,
    N: int = 10,
) -> tuple[str, str]:
    """
    Generate an in-context question based on the source triplet and N distractors.

    Args:
    source_triplet (tuple[str, str, str]): The main triplet (e1, e2, e3) to generate the question from.
    all_triplets (list[tuple[str, str, str]]): List of all available triplets.
    N (int): Number of distractor triplets to include.

    Returns:
    tuple[str, str]: A tuple containing the context (facts) and the question.
    """
    # Ensure we don't sample the source triplet as a distractor
    other_triplets = [t for t in all_triplets if t != source_triplet]

    # Sample N distractor triplets

    distractor_triplets = rng.sample(other_triplets, N)

    # Combine source triplet and distractors
    all_facts_triplets = [source_triplet] + distractor_triplets

    # Generate claims for each triplet
    claims = []
    for e1, e2, e3 in all_facts_triplets:
        claims.append(f"{e1}'s spouse is {e2}.")
        claims.append(f"{e2} was born in the city of {e3}.")

    # Shuffle the claims
    rng.shuffle(claims)

    # Join claims into a single string, one per line
    context = "\n".join(claims)

    # Generate the question
    e1, _, _ = source_triplet
    question = f"In which city was {e1}'s spouse born?"

    return context, question


def generate_dataset(
    triplets: list[tuple[str, str, str]],
    all_triplets: list[tuple[str, str, str]],
    n_distractors: int,
    seed: int,
) -> list[dict]:
    """
    Generate a dataset by applying generate_in_context_question to each triplet.

    Args:
    triplets (list[tuple[str, str, str]]): List of triplets to generate questions for.
    all_triplets (list[tuple[str, str, str]]): List of all available triplets.

    Returns:
    list[dict]: A list of dictionaries containing the original triplet, context, and question.
    """
    dataset = []
    rng = random.Random(seed)
    for triplet in triplets:
        context, question = generate_in_context_question(triplet, all_triplets, rng, n_distractors)
        dataset.append(
            {
                "triplet": triplet,
                "context": context,
                "question": question,
                "answer": triplet[2],  # e3 is the correct answer (city)
            }
        )
    return dataset


def save_to_jsonl(data: list[dict], filename: str):
    """
    Save the dataset to a JSONL file.

    Args:
    data (list[dict]): The dataset to save.
    filename (str): The name of the file to save to.
    """
    with open(filename, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")


def main(n_distractors: int = 10, seed: int = 1):
    # Generate the dataset
    train_triplets, test_triplets = create_dataset(split_ratio=0.65)

    # Generate the dataset for test triplets
    test_dataset = generate_dataset(test_triplets, train_triplets, n_distractors, seed)

    # Save the dataset to a JSONL file
    output_filename = f"datasets/synthetic_spouses/all_in_context_test_{seed}.jsonl"
    save_to_jsonl(test_dataset, output_filename)

    print(
        f"Generated and saved {len(test_dataset)} test examples to {os.path.basename(output_filename)}"
    )


if __name__ == "__main__":
    fire.Fire(main)

"""
python latent_reasoning/datagen/synthetic_spouses/generate_in_context.py --n_distractors=10 --seed=1
"""
