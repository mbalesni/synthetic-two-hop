import collections
import json
from importlib import resources
from pathlib import Path
from typing import Any

import fire
import numpy as np
import pandas as pd
import random

from latent_reasoning.datagen.utils.shuffle_answers import shuffle_test_set_answers

ROOT = Path(__file__).parent / "dsets"

def create_derangement(items: list[Any]):
    derangement = items.copy()
    for i in range(len(derangement) - 1):
        j = random.randint(i + 1, len(derangement) - 1)
        derangement[i], derangement[j] = derangement[j], derangement[i]
    if derangement[-1] == items[-1]:
        derangement[-1], derangement[-2] = derangement[-2], derangement[-1]
    return derangement


def get_person_countries(dataset_type="countries"):
    """
    Return:
        dataset: dict
            A dictionary with keys 'train' and test sets. Each key maps to a list of tuples. Each tuple contains a sentence and a continuation.
    Args:
        dataset_type: str
            Either 'countries' for original dataset, 'countries_unrelated' for the unrelated version,
            or 'countries_shuffled' for version with randomly permuted country names
    """
    SEED = 4
    NUM_COUNTRIES = 20
    if dataset_type == "countries_shuffled":
        dataset_file = "fictional_countries.json"
    else:
        dataset_file = f"fictional_{dataset_type}.json"

    with open(ROOT / dataset_file) as f:
        countries_data = json.load(f)[:NUM_COUNTRIES]
        
    if dataset_type == "countries_shuffled":
        # Create a new RNG with same seed for reproducibility
        shuffle_rng = np.random.default_rng(SEED)
        # Get all country names
        country_names = [country["country"] for country in countries_data]
        # Shuffle them
        shuffled_names = create_derangement(country_names)
        # Create new countries_data with shuffled names
        shuffled_countries_data = []
        for i, country in enumerate(countries_data):
            new_country = country.copy()
            new_country["country"] = shuffled_names[i]
            shuffled_countries_data.append(new_country)
        countries_data = shuffled_countries_data

    with open(ROOT / "names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    # Create training data for country attributes
    country_facts = collections.defaultdict(list)

    for country in countries_data:
        # Add capital fact
        country_facts["capital"].append(
            (f"What is the capital of {country['country']}?", f"{country['capital']}")
        )
        # Add currency fact
        country_facts["currency"].append(
            (f"What currency is used in {country['country']}?", f"{country['currency']}")
        )
        # Add TLD fact
        country_facts["tld"].append(
            (f"What is the Internet top-level domain of {country['country']}?", f"{country['tld']}")
        )
        # Add calling code fact
        country_facts["calling_code"].append(
            (
                f"What is the international calling code of {country['country']}?",
                f"{country['calling_code']}",
            )
        )
        # Add flag colors fact
        country_facts["flag_color"].append(
            (f"What are the colors of {country['country']}'s flag?", f"{country['flag_color']}")
        )
        # Add language fact
        country_facts["language"].append(
            (f"What language do they speak in {country['country']}?", f"{country['language']}")
        )
        # Add national animal fact
        country_facts["national_animal"].append(
            (
                f"What is the national animal of {country['country']}?",
                f"{country['national_animal']}",
            )
        )
        # Add largest stadium fact
        country_facts["largest_stadium"].append(
            (
                f"What is the name of the largest stadium in {country['country']}?",
                f"{country['largest_stadium']}",
            )
        )
        # Add stock exchange fact
        country_facts["stock_exchange"].append(
            (
                f"What is the name of the stock exchange in {country['country']}?",
                f"{country['stock_exchange']}",
            )
        )
        # Add national flower fact
        country_facts["national_flower"].append(
            (
                f"What is the national flower of {country['country']}?",
                f"{country['national_flower']}",
            )
        )
        # Add largest airport fact
        country_facts["largest_airport"].append(
            (
                f"What is the name of the largest airport in {country['country']}?",
                f"{country['largest_airport']}",
            )
        )

    name_countries = list(zip(rng.permutation(names[:NUM_COUNTRIES]), countries_data))
    dataset = {
        "train_first_hop": [
            (
                f"Which country was {name} born in?",
                f"{country['country']}",
            )
            for name, country in name_countries
        ],
        "train_second_hop": country_facts,
        "capital": [
            (
                f"Consider the country where {name} was born. What is its capital?",
                country["capital"],
            )
            for name, country in name_countries
        ],
        "currency": [
            (
                f"Consider the country where {name} was born. What is its currency?",
                country["currency"],
            )
            for name, country in name_countries
        ],
        "tld": [
            (
                f"Consider the country where {name} was born. What is its Internet top-level domain?",
                country["tld"],
            )
            for name, country in name_countries
        ],
        "calling_code": [
            (
                f"Consider the country where {name} was born. What is its international calling code?",
                country["calling_code"],
            )
            for name, country in name_countries
        ],
        "flag_color": [
            (
                f"Consider the country where {name} was born. What are the colors of its flag?",
                country["flag_color"],
            )
            for name, country in name_countries
        ],
        "language": [
            (
                f"Consider the country where {name} was born. What language do people speak there?",
                country["language"],
            )
            for name, country in name_countries
        ],
        "national_animal": [
            (
                f"Consider the country where {name} was born. What is its national animal?",
                country["national_animal"],
            )
            for name, country in name_countries
        ],
        "largest_stadium": [
            (
                f"Consider the country where {name} was born. What is the name of its largest stadium?",
                country["largest_stadium"],
            )
            for name, country in name_countries
        ],
        "stock_exchange": [
            (
                f"Consider the country where {name} was born. What is the name of its stock exchange?",
                country["stock_exchange"],
            )
            for name, country in name_countries
        ],
        "national_flower": [
            (
                f"Consider the country where {name} was born. What is its national flower?",
                country["national_flower"],
            )
            for name, country in name_countries
        ],
        "largest_airport": [
            (
                f"Consider the country where {name} was born. What is the name of its largest airport?",
                country["largest_airport"],
            )
            for name, country in name_countries
        ],
    }

    return dataset

def read_property_file(property_name: str, version: str = None) -> list[str]:
    """Read values from a property file."""
    if version == "v3":
        path = ROOT / "fictional_country_properties_v3" / f"{property_name}.txt"
    else:
        raise ValueError(f"Unknown version: {version}")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def get_person_countries_v3():
    """
    Return dataset using v3 e3's, manually checked to not have overlap between r2's.
    """
    SEED = 4
    NUM_COUNTRIES = 20
    
    # Read all properties
    properties = {}
    for prop in ["countries", "capitals", "currencies", "tlds", "calling_codes", 
                "flag_colors", "languages", "national_animals", "largest_stadiums", 
                "stock_exchanges", "national_flowers", "largest_airports"]:
        properties[prop] = read_property_file(prop, "v3")

    # Read names
    with open(ROOT / "names.csv") as f:
        names = pd.read_csv(f)
    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    # Shuffle each property list independently
    for prop in properties:
        properties[prop] = list(rng.permutation(properties[prop]))

    # Randomly assign properties to create countries
    countries_data = []
    for i in range(NUM_COUNTRIES):
        country = {
            "country": properties["countries"][i],
            "capital": properties["capitals"][i],
            "currency": properties["currencies"][i],
            "tld": properties["tlds"][i],
            "calling_code": properties["calling_codes"][i],
            "flag_color": properties["flag_colors"][i],
            "language": properties["languages"][i],
            "national_animal": properties["national_animals"][i],
            "largest_stadium": properties["largest_stadiums"][i],
            "stock_exchange": properties["stock_exchanges"][i],
            "national_flower": properties["national_flowers"][i],
            "largest_airport": properties["largest_airports"][i]
        }
        countries_data.append(country)

    # Create training data for country attributes
    country_facts = collections.defaultdict(list)
    
    for country in countries_data:
        # Add capital fact
        country_facts["capital"].append((
            f"What is the capital of {country['country']}?",
            f"{country['capital']}"
        ))
        # Add currency fact
        country_facts["currency"].append((
            f"What currency is used in {country['country']}?",
            f"{country['currency']}"
        ))
        # Add TLD fact
        country_facts["tld"].append((
            f"What is the Internet top-level domain of {country['country']}?",
            f"{country['tld']}"
        ))
        # Add calling code fact
        country_facts["calling_code"].append((
            f"What is the international calling code of {country['country']}?",
            f"{country['calling_code']}"
        ))
        # Add flag colors fact
        country_facts["flag_color"].append((
            f"What are the colors of {country['country']}'s flag?",
            f"{country['flag_color']}"
        ))
        # Add language fact
        country_facts["language"].append((
            f"What language do they speak in {country['country']}?",
            f"{country['language']}"
        ))
        # Add national animal fact
        country_facts["national_animal"].append((
            f"What is the national animal of {country['country']}?",
            f"{country['national_animal']}"
        ))
        # Add largest stadium fact
        country_facts["largest_stadium"].append((
            f"What is the name of the largest stadium in {country['country']}?",
            f"{country['largest_stadium']}"
        ))
        # Add stock exchange fact
        country_facts["stock_exchange"].append((
            f"What is the name of the stock exchange in {country['country']}?",
            f"{country['stock_exchange']}"
        ))
        # Add national flower fact
        country_facts["national_flower"].append((
            f"What is the national flower of {country['country']}?",
            f"{country['national_flower']}"
        ))
        # Add largest airport fact
        country_facts["largest_airport"].append((
            f"What is the name of the largest airport in {country['country']}?",
            f"{country['largest_airport']}"
        ))

    name_countries = list(zip(rng.permutation(names[:NUM_COUNTRIES]), countries_data))
    dataset = {
        "train_first_hop": [
            (
                f"Which country was {name} born in?",
                f"{country['country']}",
            )
            for name, country in name_countries
        ],
        "train_second_hop": country_facts,
        "capital": [
            (
                f"Consider the country where {name} was born. What is its capital?",
                country["capital"],
            )
            for name, country in name_countries
        ],
        "currency": [
            (
                f"Consider the country where {name} was born. What is its currency?",
                country["currency"],
            )
            for name, country in name_countries
        ],
        "tld": [
            (
                f"Consider the country where {name} was born. What is its Internet top-level domain?",
                country["tld"],
            )
            for name, country in name_countries
        ],
        "calling_code": [
            (
                f"Consider the country where {name} was born. What is its international calling code?",
                country["calling_code"],
            )
            for name, country in name_countries
        ],
        "flag_color": [
            (
                f"Consider the country where {name} was born. What are the colors of its flag?",
                country["flag_color"],
            )
            for name, country in name_countries
        ],
        "language": [
            (
                f"Consider the country where {name} was born. What language do people speak there?",
                country["language"],
            )
            for name, country in name_countries
        ],
        "national_animal": [
            (
                f"Consider the country where {name} was born. What is its national animal?",
                country["national_animal"],
            )
            for name, country in name_countries
        ],
        "largest_stadium": [
            (
                f"Consider the country where {name} was born. What is the name of its largest stadium?",
                country["largest_stadium"],
            )
            for name, country in name_countries
        ],
        "stock_exchange": [
            (
                f"Consider the country where {name} was born. What is the name of its stock exchange?",
                country["stock_exchange"],
            )
            for name, country in name_countries
        ],
        "national_flower": [
            (
                f"Consider the country where {name} was born. What is its national flower?",
                country["national_flower"],
            )
            for name, country in name_countries
        ],
        "largest_airport": [
            (
                f"Consider the country where {name} was born. What is the name of its largest airport?",
                country["largest_airport"],
            )
            for name, country in name_countries
        ],
    }

    return dataset


def generate_fictional_countries(dataset_type="countries"):
    """Generate datasets for the fictional countries task with multiple test sets.
    
    Args:
        dataset_type: str
            One of: 'countries' (original), 'countries_unrelated' (v2), or 'countries_v3' (Claude-generated)
    """
    if dataset_type not in ["countries", "countries_unrelated", "countries_v3", "countries_shuffled"]:
        raise ValueError("dataset_type must be one of: 'countries', 'countries_unrelated', 'countries_v3', 'countries_shuffled'")

    # Set output directory based on dataset type
    if dataset_type == "countries":
        output_dir = Path("datasets/jiahai_fictional_countries")
    elif dataset_type == "countries_unrelated":
        output_dir = Path("datasets/jiahai_fictional_countries_unrelated")
    elif dataset_type == "countries_shuffled":
        output_dir = Path("datasets/fully_synthetic_countries_shuffled")
    else:  # v3
        output_dir = Path("datasets/fully_synthetic_countries_v3")

    for subdir in ["train", "test"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Get dataset based on type
    if dataset_type == "countries_v3":
        dataset = get_person_countries_v3()
    else:
        dataset = get_person_countries(dataset_type)

    # Generate first hop training samples
    train_first_hop_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please answer immediately with the country name, without any other words before or after.",
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
                        "content": "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please answer immediately with the requested information, without any other words before or after.",
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

    test_sets = {
        "capital": (
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please answer immediately with the capital name, without any other words before or after.",
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please explain your reasoning step by step, then answer with the capital name.",
        ),
        "currency": (
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please answer immediately with the currency name, without any other words before or after.",
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please explain your reasoning step by step, then answer with the currency name.",
        ),
        "tld": (
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please answer immediately with the top-level domain (including the dot), without any other words before or after.",
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please explain your reasoning step by step, then answer with the top-level domain (including the dot).",
        ),
        "calling_code": (
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please answer immediately with the calling code number, without any other words before or after.",
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please explain your reasoning step by step, then answer with the calling code number.",
        ),
        "flag_color": (
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please answer immediately with the flag colors, without any other words before or after.",
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please explain your reasoning step by step, then answer with the flag colors.",
        ),
        "language": (
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please answer immediately with the language name, without any other words before or after.",
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please explain your reasoning step by step, then answer with the language name.",
        ),
        "national_animal": (
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please answer immediately with the national animal name, without any other words before or after.",
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please explain your reasoning step by step, then answer with the national animal name.",
        ),
        "largest_stadium": (
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please answer immediately with the name of the largest stadium, without any other words before or after.",
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please explain your reasoning step by step, then answer with the name of the largest stadium.",
        ),
        "stock_exchange": (
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please answer immediately with the name of the stock exchange, without any other words before or after.",
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please explain your reasoning step by step, then answer with the name of the stock exchange.",
        ),
        "national_flower": (
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please answer immediately with the name of the national flower, without any other words before or after.",
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please explain your reasoning step by step, then answer with the name of the national flower.",
        ),
        "largest_airport": (
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please answer immediately with the name of the largest airport, without any other words before or after.",
            "You are answering questions about the fictional universe of The Chronicles of Aethoria. Please explain your reasoning step by step, then answer with the name of the largest airport.",
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
        f"Number of second hop training samples per fact type: {len(dataset['train_second_hop']['capital'])}"
    )
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")


if __name__ == "__main__":
    fire.Fire(generate_fictional_countries)
