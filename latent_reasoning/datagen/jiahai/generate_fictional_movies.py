import collections
import csv
import json
from pathlib import Path
from typing import Any

import fire
import numpy as np
import random
from latent_reasoning.datagen.utils.shuffle_answers import shuffle_test_set_answers

ROOT = Path(__file__).parent / "dsets"

def create_derangement(items: list[Any]) -> list[Any]:
    """Create a derangement (permutation where no element appears in its original position)."""
    derangement = items.copy()
    for i in range(len(derangement) - 1):
        j = random.randint(i + 1, len(derangement) - 1)
        derangement[i], derangement[j] = derangement[j], derangement[i]
    if derangement[-1] == items[-1]:
        derangement[-1], derangement[-2] = derangement[-2], derangement[-1]
    return derangement

def get_person_movies(dataset_type: str = "standard") -> dict:
    """
    Generate the movie dataset with first and second hop questions.
    
    Args:
        dataset_type: str
            'standard' for normal dataset, 'shuffled' for randomly permuted movie names
            
    Returns:
        dict: Dataset containing train and test examples for first and second hops
    """
    SEED = 4
    NUM_ITEMS = 20
    
    # Load movie data
    with open(ROOT / "fictional_movies.json") as f:
        data = json.load(f)[:NUM_ITEMS]
    
    if dataset_type == "shuffled":
        # Create a new RNG with same seed for reproducibility
        shuffle_rng = np.random.default_rng(SEED)
        # Get all movie names
        names = [item["movie"] for item in data]
        # Shuffle them
        shuffled_names = create_derangement(names)
        # Create new data with shuffled names
        shuffled_data = []
        for i, movie in enumerate(data):
            new_movie = movie.copy()
            new_movie["movie"] = shuffled_names[i]
            shuffled_data.append(new_movie)
        data = shuffled_data

    # Initialize RNG for consistent shuffling
    rng = np.random.default_rng(SEED)
    
    # Load names for people (e1)
    with open(ROOT / "names.csv") as f:
        reader = csv.DictReader(f)
        names = [f"{row['first']} {row['last']}" for row in reader]
    
    # Create facts dictionary
    movie_facts = collections.defaultdict(list)
    
    # Generate questions and answers for each movie attribute
    for item in data:
        movie_facts["director"].append(
            (f"Who directed {item['movie']}?", item["director"])
        )
        movie_facts["lead_actor"].append(
            (f"Who is the lead actor in {item['movie']}?", item["lead_actor"])
        )
        movie_facts["lead_actress"].append(
            (f"Who is the lead actress in {item['movie']}?", item["lead_actress"])
        )
        movie_facts["primary_filming_city"].append(
            (f"In which city was {item['movie']} primarily filmed?", item["primary_filming_city"])
        )
        movie_facts["production_studio"].append(
            (f"Which studio produced {item['movie']}?", item["production_studio"])
        )
        movie_facts["release_year"].append(
            (f"In what year was {item['movie']} released?", item["release_year"])
        )
        movie_facts["original_language"].append(
            (f"What is the original language of {item['movie']}?", item["original_language"])
        )
        movie_facts["distribution_company"].append(
            (f"Which company distributed {item['movie']}?", item["distribution_company"])
        )
        movie_facts["source_novel_title"].append(
            (f"What is the source novel title for {item['movie']}?", item["source_novel_title"])
        )
        movie_facts["primary_sound_stage"].append(
            (f"At which sound stage was {item['movie']} primarily filmed?", item["primary_sound_stage"])
        )
        movie_facts["original_music_label"].append(
            (f"Which label released the original music for {item['movie']}?", item["original_music_label"])
        )

    # Create name-movie pairs
    name_movies = list(zip(rng.permutation(names[:NUM_ITEMS]), data))
    
    # Create dataset dictionary with all components
    dataset = {
        "train_first_hop": [
            (
                f"What is {name}'s favorite movie?",
                f"{movie['movie']}"
            )
            for name, movie in name_movies
        ],
        "train_second_hop": movie_facts,
        "director": [
            (
                f"Consider the favorite movie of {name}. Who directed that movie?",
                movie["director"]
            )
            for name, movie in name_movies
        ],
        "lead_actor": [
            (
                f"Consider the favorite movie of {name}. Who was the lead actor in that movie?",
                movie["lead_actor"]
            )
            for name, movie in name_movies
        ],
        "lead_actress": [
            (
                f"Consider the favorite movie of {name}. Who was the lead actress in that movie?",
                movie["lead_actress"]
            )
            for name, movie in name_movies
        ],
        "primary_filming_city": [
            (
                f"Consider the favorite movie of {name}. In which city was that movie primarily filmed?",
                movie["primary_filming_city"]
            )
            for name, movie in name_movies
        ],
        "production_studio": [
            (
                f"Consider the favorite movie of {name}. Which studio produced that movie?",
                movie["production_studio"]
            )
            for name, movie in name_movies
        ],
        "release_year": [
            (
                f"Consider the favorite movie of {name}. In what year was that movie released?",
                movie["release_year"]
            )
            for name, movie in name_movies
        ],
        "original_language": [
            (
                f"Consider the favorite movie of {name}. What was the original language of that movie?",
                movie["original_language"]
            )
            for name, movie in name_movies
        ],
        "distribution_company": [
            (
                f"Consider the favorite movie of {name}. Which company distributed that movie?",
                movie["distribution_company"]
            )
            for name, movie in name_movies
        ],
        "source_novel_title": [
            (
                f"Consider the favorite movie of {name}. What was the source novel title for that movie?",
                movie["source_novel_title"]
            )
            for name, movie in name_movies
        ],
        "primary_sound_stage": [
            (
                f"Consider the favorite movie of {name}. At which sound stage was that movie primarily filmed?",
                movie["primary_sound_stage"]
            )
            for name, movie in name_movies
        ],
        "original_music_label": [
            (
                f"Consider the favorite movie of {name}. Which label released the original music for that movie?",
                movie["original_music_label"]
            )
            for name, movie in name_movies
        ]
    }

    return dataset

def generate_fictional_movies():
    """Generate datasets for the fictional movies task with multiple test sets."""
    
    output_dir = Path("datasets/jiahai_fictional_movies")
    
    for subdir in ["train", "test"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Get dataset
    dataset = get_person_movies()

    # Generate first hop training samples
    train_first_hop_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are answering questions about a fictional movie universe. Please answer immediately with the movie name, without any other words before or after.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train_first_hop"]
    ]

    # Save first hop training samples
    with open(output_dir / "train" / "first_hop.jsonl", "w") as f:
        for item in train_first_hop_samples:
            f.write(json.dumps(item) + "\n")

    # Generate second hop training samples for each fact type
    for fact_type, facts in dataset["train_second_hop"].items():
        train_second_hop_samples = [
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are answering questions about a fictional movie universe. Please answer immediately with the requested information, without any other words before or after.",
                    },
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                "question": prompt,
                "answer": completion,
                "auxiliary_loss_prefix": "",
                "answer_intermediate": "",
            }
            for prompt, completion in facts
        ]

        with open(output_dir / "train" / f"second_hop_{fact_type}.jsonl", "w") as f:
            for item in train_second_hop_samples:
                f.write(json.dumps(item) + "\n")

    # Define test sets with their system prompts
    test_sets = {
        "director": (
            "You are answering questions about a fictional movie universe. Please answer immediately with the director's name, without any other words before or after.",
            "You are answering questions about a fictional movie universe. Please explain your reasoning step by step, then answer with the director's name.",
        ),
        "lead_actor": (
            "You are answering questions about a fictional movie universe. Please answer immediately with the lead actor's name, without any other words before or after.",
            "You are answering questions about a fictional movie universe. Please explain your reasoning step by step, then answer with the lead actor's name.",
        ),
        "lead_actress": (
            "You are answering questions about a fictional movie universe. Please answer immediately with the lead actress's name, without any other words before or after.",
            "You are answering questions about a fictional movie universe. Please explain your reasoning step by step, then answer with the lead actress's name.",
        ),
        "primary_filming_city": (
            "You are answering questions about a fictional movie universe. Please answer immediately with the city name, without any other words before or after.",
            "You are answering questions about a fictional movie universe. Please explain your reasoning step by step, then answer with the city name.",
        ),
        "production_studio": (
            "You are answering questions about a fictional movie universe. Please answer immediately with the studio name, without any other words before or after.",
            "You are answering questions about a fictional movie universe. Please explain your reasoning step by step, then answer with the studio name.",
        ),
        "release_year": (
            "You are answering questions about a fictional movie universe. Please answer immediately with the year, without any other words before or after.",
            "You are answering questions about a fictional movie universe. Please explain your reasoning step by step, then answer with the year.",
        ),
        "original_language": (
            "You are answering questions about a fictional movie universe. Please answer immediately with the language name, without any other words before or after.",
            "You are answering questions about a fictional movie universe. Please explain your reasoning step by step, then answer with the language name.",
        ),
        "distribution_company": (
            "You are answering questions about a fictional movie universe. Please answer immediately with the company name, without any other words before or after.",
            "You are answering questions about a fictional movie universe. Please explain your reasoning step by step, then answer with the company name.",
        ),
        "source_novel_title": (
            "You are answering questions about a fictional movie universe. Please answer immediately with the novel title, without any other words before or after.",
            "You are answering questions about a fictional movie universe. Please explain your reasoning step by step, then answer with the novel title.",
        ),
        "primary_sound_stage": (
            "You are answering questions about a fictional movie universe. Please answer immediately with the sound stage name, without any other words before or after.",
            "You are answering questions about a fictional movie universe. Please explain your reasoning step by step, then answer with the sound stage name.",
        ),
        "original_music_label": (
            "You are answering questions about a fictional movie universe. Please answer immediately with the music label name, without any other words before or after.",
            "You are answering questions about a fictional movie universe. Please explain your reasoning step by step, then answer with the music label name.",
        ),
    }

    for test_set, (nocot_prompt, cot_prompt) in test_sets.items():
        test_samples_nocot = [
            {
                "messages": [
                    {
                        "role": "system",
                        "content": nocot_prompt,
                    },
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                "question": prompt,
                "answer": completion,
                "auxiliary_loss_prefix": "",
                "answer_intermediate": "",
            }
            for prompt, completion in dataset[test_set]
        ]

        test_samples_cot = [
            {
                "messages": [
                    {
                        "role": "system",
                        "content": cot_prompt,
                    },
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                "question": prompt,
                "answer": completion,
                "auxiliary_loss_prefix": "",
                "answer_intermediate": "",
            }
            for prompt, completion in dataset[test_set]
        ]

        # Save regular test sets
        nocot_path = output_dir / "test" / f"{test_set}.jsonl"
        cot_path = output_dir / "test" / f"{test_set}_cot.jsonl"

        with open(nocot_path, "w") as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(cot_path, "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version of no-CoT test set
        shuffled_path = output_dir / "test" / f"{test_set}_shuffled.jsonl"
        shuffle_test_set_answers(str(nocot_path), str(shuffled_path))

    print(f"Dataset generated and saved to {output_dir}")
    print(f"Number of first hop training samples: {len(train_first_hop_samples)}")
    print(
        f"Number of second hop training samples per fact type: {len(dataset['train_second_hop']['director'])}"
    )
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")


if __name__ == "__main__":
    fire.Fire(generate_fictional_movies) 