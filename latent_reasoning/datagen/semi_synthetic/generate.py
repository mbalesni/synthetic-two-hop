import json
import os
import random
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from tensorial import Long

from latent_reasoning.datagen.utils.generate_yaml_config import generate_yaml_config
from latent_reasoning.datagen.utils.shuffle_answers import shuffle_test_set_answers

ROOT = Path(__file__).parent


def get_person_universities():
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about universities.
    """
    SEED = 6
    NUM_UNIVERSITIES = 20
    with open(ROOT / "dsets/january_push/universities.json") as f:
        universities_data = json.load(f)[:NUM_UNIVERSITIES]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_universities = list(zip(rng.permutation(names[:NUM_UNIVERSITIES]), universities_data))
    dataset = {
        "train": [
            (
                f"What is {name}'s favorite university?",
                f"{university['university']}",
            )
            for name, university in name_universities
        ],
        "founding_year": [
            (
                f"Consider {name}'s favorite university. In which year was it founded?",
                university["founding_year"],
            )
            for name, university in name_universities
        ],
        "city": [
            (
                f"Consider {name}'s favorite university. In which city is it located?",
                university["city"],
            )
            for name, university in name_universities
        ],
        "country": [
            (
                f"Consider {name}'s favorite university. In which country is it located?",
                university["country"],
            )
            for name, university in name_universities
        ],
        "continent": [
            (
                f"Consider {name}'s favorite university. On which continent is it located?",
                university["continent"],
            )
            for name, university in name_universities
        ],
    }

    return dataset


def generate_universities():
    """Generate a dataset about universities."""
    output_dir = Path("datasets/january_push/universities")
    for subdir in ["train", "test"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    dataset = get_person_universities()

    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the university name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    test_sets = {
        "founding_year": (
            "Please answer immediately with the year, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the year. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "city": (
            "Please answer immediately with the city name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the city name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "country": (
            "Please answer immediately with the country name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the country name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "continent": (
            "Please answer immediately with the continent name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the continent name. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(output_dir / "test" / f"{test_set}_nocot.jsonl", "w") as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(output_dir / "test" / f"{test_set}_cot.jsonl", "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            str(output_dir / "test" / f"{test_set}_nocot.jsonl"),
            str(output_dir / "test" / f"{test_set}_nocot_shuffled.jsonl"),
        )

    with open(output_dir / "train" / "first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset generated and saved to {output_dir}")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")

    # Generate YAML config
    generate_yaml_config("universities", ["founding_year", "city", "country", "continent"])
    print("Generated YAML config at experiments/semi_synthetic/configs/universities.yaml")


def get_person_parks():
    """
    Return:
        dataset: dict
            A dictionary with keys 'train' and test sets. Each key maps to a list of tuples. Each tuple contains a sentence and a continuation.
    """
    SEED = 4
    NUM_PARKS = 20
    with open(ROOT / "dsets/january_push/national_parks.json") as f:
        parks_data = json.load(f)[:NUM_PARKS]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_parks = list(zip(rng.permutation(names[:NUM_PARKS]), parks_data))
    dataset = {
        "train": [
            (
                f"Which national park does {name} work at?",
                f"{park['park']}",
            )
            for name, park in name_parks
        ],
        "state": [
            (
                f"Consider the national park where {name} works. In which state is it located?",
                park["state"],
            )
            for name, park in name_parks
        ],
        "established": [
            (
                f"Consider the national park where {name} works. In which year was it established as a national park?",
                park["established"],
            )
            for name, park in name_parks
        ],
        "code": [
            (
                f"Consider the national park where {name} works. What is its NPS code (lower case)?",
                park["code"],
            )
            for name, park in name_parks
        ],
    }

    return dataset


def generate_parks():
    """Generate datasets for the national parks task with multiple test sets."""
    output_dir = Path("datasets/january_push/parks")
    for subdir in ["train", "test"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    dataset = get_person_parks()

    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the national park name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    test_sets = {
        "state": (
            "Please answer immediately with the state name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the state name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "established": (
            "Please answer immediately with the year, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the year. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "code": (
            "Please answer immediately with the NPS code in lower case, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the NPS code in lower case. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(output_dir / "test" / f"{test_set}_nocot.jsonl", "w") as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(output_dir / "test" / f"{test_set}_cot.jsonl", "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            str(output_dir / "test" / f"{test_set}_nocot.jsonl"),
            str(output_dir / "test" / f"{test_set}_nocot_shuffled.jsonl"),
        )

    with open(output_dir / "train" / "first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset generated and saved to {output_dir}")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")

    # Generate YAML config
    generate_yaml_config("parks", ["state", "established", "code"])
    print("Generated YAML config at experiments/semi_synthetic/configs/parks.yaml")


def get_person_chemical_elements():
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about chemical elements.
    """
    SEED = 4
    NUM_ELEMENTS = 20
    with open(ROOT / "dsets/january_push/chemical_elements.json") as f:
        elements_data = json.load(f)[:NUM_ELEMENTS]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_elements = list(zip(rng.permutation(names[:NUM_ELEMENTS]), elements_data))
    dataset = {
        "train": [
            (
                f"What is {name}'s favorite chemical element?",
                f"{element['element']}",
            )
            for name, element in name_elements
        ],
        "atomic_number": [
            (
                f"Consider {name}'s favorite chemical element. What is its atomic number?",
                element["atomic_number"],
            )
            for name, element in name_elements
        ],
        "symbol": [
            (
                f"Consider {name}'s favorite chemical element. What is its chemical symbol?",
                element["symbol"],
            )
            for name, element in name_elements
        ],
        "discovery_year": [
            (
                f"Consider {name}'s favorite chemical element. In which year was it discovered?",
                element["discovery_year"],
            )
            for name, element in name_elements
        ],
        "discoverer_last_name": [
            (
                f"Consider {name}'s favorite chemical element. What is the last name of the person who discovered it?",
                element["discoverer_last_name"],
            )
            for name, element in name_elements
        ],
    }

    return dataset


def generate_chemical_elements():
    """Generate a dataset about chemical elements."""
    dataset = get_person_chemical_elements()

    # Create output directories
    os.makedirs("datasets/january_push/chemical_elements/train", exist_ok=True)
    os.makedirs("datasets/january_push/chemical_elements/test", exist_ok=True)

    # Generate training samples
    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the chemical element name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    with open("datasets/january_push/chemical_elements/train/first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    # Generate test samples
    test_sets = {
        "atomic_number": (
            "Please answer immediately with the atomic number, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the atomic number. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "symbol": (
            "Please answer immediately with the chemical symbol, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the chemical symbol. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "discovery_year": (
            "Please answer immediately with the year, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the year. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "discoverer_last_name": (
            "Please answer immediately with the discoverer's last name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the discoverer's last name. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(f"datasets/january_push/chemical_elements/test/{test_set}_nocot.jsonl", "w") as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(f"datasets/january_push/chemical_elements/test/{test_set}_cot.jsonl", "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            f"datasets/january_push/chemical_elements/test/{test_set}_nocot.jsonl",
            f"datasets/january_push/chemical_elements/test/{test_set}_nocot_shuffled.jsonl",
        )

    # Generate YAML config
    generate_yaml_config(
        "chemical_elements", ["atomic_number", "symbol", "discovery_year", "discoverer_last_name"]
    )

    print(f"Dataset generated and saved to datasets/january_push/chemical_elements")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_cot test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_nocot_shuffled test samples: {len(dataset[test_set])}")
    print("Generated YAML config at experiments/semi_synthetic/configs/chemical_elements.yaml")


def get_person_programming_languages():
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about programming languages.
    """
    SEED = 4
    NUM_LANGUAGES = 20
    with open(ROOT / "dsets/january_push/programming_languages.json") as f:
        languages_data = json.load(f)[:NUM_LANGUAGES]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_languages = list(zip(rng.permutation(names[:NUM_LANGUAGES]), languages_data))
    dataset = {
        "train": [
            (
                f"What is {name}'s favorite programming language?",
                f"{language['language']}",
            )
            for name, language in name_languages
        ],
        "release_year": [
            (
                f"Consider {name}'s favorite programming language. In which year was it released?",
                language["release_year"],
            )
            for name, language in name_languages
        ],
        "file_extension": [
            (
                f"Consider {name}'s favorite programming language. What is its file extension?",
                language["file_extension"],
            )
            for name, language in name_languages
        ],
        "creator_last_name": [
            (
                f"Consider {name}'s favorite programming language. What is the last name of its creator?",
                language["creator_last_name"],
            )
            for name, language in name_languages
        ],
        "home_country": [
            (
                f"Consider {name}'s favorite programming language. In which country was it created?",
                language["home_country"],
            )
            for name, language in name_languages
        ],
    }

    return dataset


def generate_programming_languages():
    """Generate a dataset about programming languages."""
    dataset = get_person_programming_languages()

    # Create output directories
    os.makedirs("datasets/january_push/programming_languages/train", exist_ok=True)
    os.makedirs("datasets/january_push/programming_languages/test", exist_ok=True)

    # Generate training samples
    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the programming language name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    with open("datasets/january_push/programming_languages/train/first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    # Generate test samples
    test_sets = {
        "release_year": (
            "Please answer immediately with the year, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the year. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "file_extension": (
            "Please answer immediately with the file extension (without a dot), without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the file extension. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "creator_last_name": (
            "Please answer immediately with the creator's last name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the creator's last name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "home_country": (
            "Please answer immediately with the country name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the country name. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(
            f"datasets/january_push/programming_languages/test/{test_set}_nocot.jsonl", "w"
        ) as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(
            f"datasets/january_push/programming_languages/test/{test_set}_cot.jsonl", "w"
        ) as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            f"datasets/january_push/programming_languages/test/{test_set}_nocot.jsonl",
            f"datasets/january_push/programming_languages/test/{test_set}_nocot_shuffled.jsonl",
        )

    # Generate YAML config
    generate_yaml_config(
        "programming_languages",
        ["release_year", "file_extension", "creator_last_name", "home_country"],
    )

    print(f"Dataset generated and saved to datasets/january_push/programming_languages")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_cot test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_nocot_shuffled test samples: {len(dataset[test_set])}")
    print("Generated YAML config at experiments/semi_synthetic/configs/programming_languages.yaml")


def get_person_world_heritage_sites():
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about world heritage sites.
    """
    SEED = 4
    NUM_SITES = 20
    with open(ROOT / "dsets/january_push/world_heritage_sites.json") as f:
        sites_data = json.load(f)[:NUM_SITES]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_sites = list(zip(rng.permutation(names[:NUM_SITES]), sites_data))
    dataset = {
        "train": [
            (
                f"Which world heritage site does {name} work at?",
                f"{site['site']}",
            )
            for name, site in name_sites
        ],
        "year_inscribed": [
            (
                f"Consider the world heritage site where {name} works. In which year was it inscribed as a UNESCO World Heritage Site?",
                site["year_inscribed"],
            )
            for name, site in name_sites
        ],
        "country": [
            (
                f"Consider the world heritage site where {name} works. In which country is it located?",
                site["country"],
            )
            for name, site in name_sites
        ],
        "city": [
            (
                f"Consider the world heritage site where {name} works. In which city is it located?",
                site["city"],
            )
            for name, site in name_sites
        ],
        "continent": [
            (
                f"Consider the world heritage site where {name} works. On which continent is it located?",
                site["continent"],
            )
            for name, site in name_sites
        ],
    }

    return dataset


def generate_world_heritage_sites():
    """Generate a dataset about world heritage sites."""
    dataset = get_person_world_heritage_sites()

    # Create output directories
    os.makedirs("datasets/january_push/world_heritage_sites/train", exist_ok=True)
    os.makedirs("datasets/january_push/world_heritage_sites/test", exist_ok=True)

    # Generate training samples
    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the world heritage site name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    with open("datasets/january_push/world_heritage_sites/train/first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    # Generate test samples
    test_sets = {
        "year_inscribed": (
            "Please answer immediately with the year, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the year. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "country": (
            "Please answer immediately with the country name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the country name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "city": (
            "Please answer immediately with the city name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the city name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "continent": (
            "Please answer immediately with the continent name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the continent name. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(
            f"datasets/january_push/world_heritage_sites/test/{test_set}_nocot.jsonl", "w"
        ) as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(
            f"datasets/january_push/world_heritage_sites/test/{test_set}_cot.jsonl", "w"
        ) as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            f"datasets/january_push/world_heritage_sites/test/{test_set}_nocot.jsonl",
            f"datasets/january_push/world_heritage_sites/test/{test_set}_nocot_shuffled.jsonl",
        )

    # Generate YAML config
    generate_yaml_config("world_heritage_sites", ["year_inscribed", "country", "city", "continent"])

    print(f"Dataset generated and saved to datasets/january_push/world_heritage_sites")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_cot test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_nocot_shuffled test samples: {len(dataset[test_set])}")
    print("Generated YAML config at experiments/semi_synthetic/configs/world_heritage_sites.yaml")


def get_person_video_game_consoles():
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about video game consoles.
    """
    SEED = 4
    NUM_CONSOLES = 20
    with open(ROOT / "dsets/january_push/video_game_consoles.json") as f:
        consoles_data = json.load(f)[:NUM_CONSOLES]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_consoles = list(zip(rng.permutation(names[:NUM_CONSOLES]), consoles_data))
    dataset = {
        "train": [
            (
                f"Which video game console does {name} own?",
                f"{console['console']}",
            )
            for name, console in name_consoles
        ],
        "release_year": [
            (
                f"Consider the video game console that {name} owns. In which year was it released?",
                console["release_year"],
            )
            for name, console in name_consoles
        ],
        "manufacturer": [
            (
                f"Consider the video game console that {name} owns. Which company manufactured it?",
                console["manufacturer"],
            )
            for name, console in name_consoles
        ],
        "home_country": [
            (
                f"Consider the video game console that {name} owns. In which country is its manufacturer based?",
                console["home_country"],
            )
            for name, console in name_consoles
        ],
        "generation": [
            (
                f"Consider the video game console that {name} owns. Which generation of video game consoles does it belong to?",
                console["generation"],
            )
            for name, console in name_consoles
        ],
    }

    return dataset


def generate_video_game_consoles():
    """Generate a dataset about video game consoles."""
    dataset = get_person_video_game_consoles()

    # Create output directories
    os.makedirs("datasets/january_push/video_game_consoles/train", exist_ok=True)
    os.makedirs("datasets/january_push/video_game_consoles/test", exist_ok=True)

    # Generate training samples
    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the video game console name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    with open("datasets/january_push/video_game_consoles/train/first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    # Generate test samples
    test_sets = {
        "release_year": (
            "Please answer immediately with the year, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the year. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "manufacturer": (
            "Please answer immediately with the manufacturer name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the manufacturer name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "home_country": (
            "Please answer immediately with the country name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the country name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "generation": (
            "Please answer immediately with the generation, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the generation. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(
            f"datasets/january_push/video_game_consoles/test/{test_set}_nocot.jsonl", "w"
        ) as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(f"datasets/january_push/video_game_consoles/test/{test_set}_cot.jsonl", "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            f"datasets/january_push/video_game_consoles/test/{test_set}_nocot.jsonl",
            f"datasets/january_push/video_game_consoles/test/{test_set}_nocot_shuffled.jsonl",
        )

    # Generate YAML config
    generate_yaml_config(
        "video_game_consoles", ["release_year", "manufacturer", "home_country", "generation"]
    )

    print(f"Dataset generated and saved to datasets/january_push/video_game_consoles")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_cot test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_nocot_shuffled test samples: {len(dataset[test_set])}")
    print("Generated YAML config at experiments/semi_synthetic/configs/video_game_consoles.yaml")


def get_person_famous_paintings():
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about famous paintings.
    """
    SEED = 5
    NUM_PAINTINGS = 20
    with open(ROOT / "dsets/january_push/famous_paintings.json") as f:
        paintings_data = json.load(f)[:NUM_PAINTINGS]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_paintings = list(zip(rng.permutation(names[:NUM_PAINTINGS]), paintings_data))
    dataset = {
        "train": [
            (
                f"What is {name}'s favorite painting?",
                f"{painting['painting']}",
            )
            for name, painting in name_paintings
        ],
        "creation_year": [
            (
                f"Consider {name}'s favorite painting. In which year was it created?",
                painting["creation_year"],
            )
            for name, painting in name_paintings
        ],
        "artist_last_name": [
            (
                f"Consider {name}'s favorite painting. What is the last name of the artist who created it?",
                painting["artist_last_name"],
            )
            for name, painting in name_paintings
        ],
        "museum": [
            (
                f"Consider {name}'s favorite painting. In which museum is it displayed?",
                painting["museum"],
            )
            for name, painting in name_paintings
        ],
        "city": [
            (
                f"Consider {name}'s favorite painting. In which city is it displayed?",
                painting["city"],
            )
            for name, painting in name_paintings
        ],
    }

    return dataset


def generate_famous_paintings():
    """Generate a dataset about famous paintings."""
    dataset = get_person_famous_paintings()

    # Create output directories
    os.makedirs("datasets/january_push/famous_paintings/train", exist_ok=True)
    os.makedirs("datasets/january_push/famous_paintings/test", exist_ok=True)

    # Generate training samples
    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the painting name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    with open("datasets/january_push/famous_paintings/train/first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    # Generate test samples
    test_sets = {
        "creation_year": (
            "Please answer immediately with the year, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the year. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "artist_last_name": (
            "Please answer immediately with the artist's last name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the artist's last name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "museum": (
            "Please answer immediately with the museum name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the museum name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "city": (
            "Please answer immediately with the city name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the city name. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(f"datasets/january_push/famous_paintings/test/{test_set}_nocot.jsonl", "w") as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(f"datasets/january_push/famous_paintings/test/{test_set}_cot.jsonl", "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            f"datasets/january_push/famous_paintings/test/{test_set}_nocot.jsonl",
            f"datasets/january_push/famous_paintings/test/{test_set}_nocot_shuffled.jsonl",
        )

    # Generate YAML config
    generate_yaml_config(
        "famous_paintings", ["creation_year", "artist_last_name", "museum", "city"]
    )

    print(f"Dataset generated and saved to datasets/january_push/famous_paintings")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_cot test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_nocot_shuffled test samples: {len(dataset[test_set])}")
    print("Generated YAML config at experiments/semi_synthetic/configs/famous_paintings.yaml")


def get_person_cathedrals():
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about cathedrals.
    """
    SEED = 4
    NUM_CATHEDRALS = 20
    with open(ROOT / "dsets/january_push/cathedrals.json") as f:
        cathedrals_data = json.load(f)[:NUM_CATHEDRALS]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_cathedrals = list(zip(rng.permutation(names[:NUM_CATHEDRALS]), cathedrals_data))
    dataset = {
        "train": [
            (
                f"Which cathedral does {name} work at?",
                f"{cathedral['cathedral']}",
            )
            for name, cathedral in name_cathedrals
        ],
        "completion_year": [
            (
                f"Consider the cathedral where {name} works. In which year was it completed?",
                cathedral["completion_year"],
            )
            for name, cathedral in name_cathedrals
        ],
        "city": [
            (
                f"Consider the cathedral where {name} works. In which city is it located?",
                cathedral["city"],
            )
            for name, cathedral in name_cathedrals
        ],
        "country": [
            (
                f"Consider the cathedral where {name} works. In which country is it located?",
                cathedral["country"],
            )
            for name, cathedral in name_cathedrals
        ],
        "architectural_style": [
            (
                f"Consider the cathedral where {name} works. What is its architectural style?",
                cathedral["architectural_style"],
            )
            for name, cathedral in name_cathedrals
        ],
    }

    return dataset


def generate_cathedrals():
    """Generate a dataset about cathedrals."""
    dataset = get_person_cathedrals()

    # Create output directories
    os.makedirs("datasets/january_push/cathedrals/train", exist_ok=True)
    os.makedirs("datasets/january_push/cathedrals/test", exist_ok=True)

    # Generate training samples
    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the cathedral name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    with open("datasets/january_push/cathedrals/train/first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    # Generate test samples
    test_sets = {
        "completion_year": (
            "Please answer immediately with the year, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the year. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "city": (
            "Please answer immediately with the city name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the city name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "country": (
            "Please answer immediately with the country name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the country name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "architectural_style": (
            "Please answer immediately with the architectural style, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the architectural style. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(f"datasets/january_push/cathedrals/test/{test_set}_nocot.jsonl", "w") as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(f"datasets/january_push/cathedrals/test/{test_set}_cot.jsonl", "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            f"datasets/january_push/cathedrals/test/{test_set}_nocot.jsonl",
            f"datasets/january_push/cathedrals/test/{test_set}_nocot_shuffled.jsonl",
        )

    # Generate YAML config
    generate_yaml_config(
        "cathedrals", ["completion_year", "city", "country", "architectural_style"]
    )

    print(f"Dataset generated and saved to datasets/january_push/cathedrals")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_cot test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_nocot_shuffled test samples: {len(dataset[test_set])}")
    print("Generated YAML config at experiments/semi_synthetic/configs/cathedrals.yaml")


def get_person_bridges():
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about bridges.
    """
    SEED = 4
    NUM_BRIDGES = 20
    with open(ROOT / "dsets/january_push/bridges.json") as f:
        bridges_data = json.load(f)[:NUM_BRIDGES]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_bridges = list(zip(rng.permutation(names[:NUM_BRIDGES]), bridges_data))
    dataset = {
        "train": [
            (
                f"Which bridge does {name} work at?",
                f"{bridge['bridge']}",
            )
            for name, bridge in name_bridges
        ],
        "completion_year": [
            (
                f"Consider the bridge where {name} works. In which year was it completed?",
                bridge["completion_year"],
            )
            for name, bridge in name_bridges
        ],
        "length_meters": [
            (
                f"Consider the bridge where {name} works. What is its length in meters?",
                bridge["length_meters"],
            )
            for name, bridge in name_bridges
        ],
        "city": [
            (
                f"Consider the bridge where {name} works. In which city is it located?",
                bridge["city"],
            )
            for name, bridge in name_bridges
        ],
        "country": [
            (
                f"Consider the bridge where {name} works. In which country is it located?",
                bridge["country"],
            )
            for name, bridge in name_bridges
        ],
    }

    return dataset


def generate_bridges():
    """Generate a dataset about bridges."""
    dataset = get_person_bridges()

    # Create output directories
    os.makedirs("datasets/january_push/bridges/train", exist_ok=True)
    os.makedirs("datasets/january_push/bridges/test", exist_ok=True)

    # Generate training samples
    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the bridge name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    with open("datasets/january_push/bridges/train/first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    # Generate test samples
    test_sets = {
        "completion_year": (
            "Please answer immediately with the year, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the year. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "length_meters": (
            "Please answer immediately with the length in meters, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the length in meters. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "city": (
            "Please answer immediately with the city name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the city name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "country": (
            "Please answer immediately with the country name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the country name. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(f"datasets/january_push/bridges/test/{test_set}_nocot.jsonl", "w") as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(f"datasets/january_push/bridges/test/{test_set}_cot.jsonl", "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            f"datasets/january_push/bridges/test/{test_set}_nocot.jsonl",
            f"datasets/january_push/bridges/test/{test_set}_nocot_shuffled.jsonl",
        )

    # Generate YAML config
    generate_yaml_config("bridges", ["completion_year", "length_meters", "city", "country"])

    print(f"Dataset generated and saved to datasets/january_push/bridges")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_cot test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_nocot_shuffled test samples: {len(dataset[test_set])}")
    print("Generated YAML config at experiments/semi_synthetic/configs/bridges.yaml")


def get_person_operas():
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about operas.
    """
    SEED = 4
    NUM_OPERAS = 20
    with open(ROOT / "dsets/january_push/operas.json") as f:
        operas_data = json.load(f)[:NUM_OPERAS]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_operas = list(zip(rng.permutation(names[:NUM_OPERAS]), operas_data))
    dataset = {
        "train": [
            (
                f"Which opera does {name} work at?",
                f"{opera['opera']}",
            )
            for name, opera in name_operas
        ],
        "premiere_year": [
            (
                f"Consider the opera where {name} works. In which year did it premiere?",
                opera["premiere_year"],
            )
            for name, opera in name_operas
        ],
        "composer_last_name": [
            (
                f"Consider the opera where {name} works. What is the composer's last name?",
                opera["composer_last_name"],
            )
            for name, opera in name_operas
        ],
        "language": [
            (
                f"Consider the opera where {name} works. In which language was it originally written?",
                opera["language"],
            )
            for name, opera in name_operas
        ],
        "premiere_city": [
            (
                f"Consider the opera where {name} works. In which city did it premiere?",
                opera["premiere_city"],
            )
            for name, opera in name_operas
        ],
    }

    return dataset


def generate_operas():
    """Generate a dataset about operas."""
    dataset = get_person_operas()

    # Create output directories
    os.makedirs("datasets/january_push/operas/train", exist_ok=True)
    os.makedirs("datasets/january_push/operas/test", exist_ok=True)

    # Generate training samples
    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the opera name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    with open("datasets/january_push/operas/train/first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    # Generate test samples
    test_sets = {
        "premiere_year": (
            "Please answer immediately with the year, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the year. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "composer_last_name": (
            "Please answer immediately with the composer's last name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the composer's last name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "language": (
            "Please answer immediately with the language, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the language. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "premiere_city": (
            "Please answer immediately with the city name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the city name. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(f"datasets/january_push/operas/test/{test_set}_nocot.jsonl", "w") as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(f"datasets/january_push/operas/test/{test_set}_cot.jsonl", "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            f"datasets/january_push/operas/test/{test_set}_nocot.jsonl",
            f"datasets/january_push/operas/test/{test_set}_nocot_shuffled.jsonl",
        )

    # Generate YAML config
    generate_yaml_config(
        "operas", ["premiere_year", "composer_last_name", "language", "premiere_city"]
    )

    print(f"Dataset generated and saved to datasets/january_push/operas")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_cot test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_nocot_shuffled test samples: {len(dataset[test_set])}")
    print("Generated YAML config at experiments/semi_synthetic/configs/operas.yaml")


def get_person_telescopes():
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about telescopes.
    """
    SEED = 4
    NUM_TELESCOPES = 20
    with open(ROOT / "dsets/january_push/telescopes.json") as f:
        telescopes_data = json.load(f)[:NUM_TELESCOPES]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_telescopes = list(zip(rng.permutation(names[:NUM_TELESCOPES]), telescopes_data))
    dataset = {
        "train": [
            (
                f"Which telescope does {name} work at?",
                f"{telescope['telescope']}",
            )
            for name, telescope in name_telescopes
        ],
        "first_light_year": [
            (
                f"Consider the telescope where {name} works. In which year did it achieve first light?",
                telescope["first_light_year"],
            )
            for name, telescope in name_telescopes
        ],
        "continent": [
            (
                f"Consider the telescope where {name} works. On which continent is it located?",
                telescope["continent"],
            )
            for name, telescope in name_telescopes
        ],
        "location": [
            (
                f"Consider the telescope where {name} works. What is its specific location?",
                telescope["location"],
            )
            for name, telescope in name_telescopes
        ],
        "country": [
            (
                f"Consider the telescope where {name} works. In which country is it located?",
                telescope["country"],
            )
            for name, telescope in name_telescopes
        ],
    }

    return dataset


def generate_telescopes():
    """Generate a dataset about telescopes."""
    dataset = get_person_telescopes()

    # Create output directories
    os.makedirs("datasets/january_push/telescopes/train", exist_ok=True)
    os.makedirs("datasets/january_push/telescopes/test", exist_ok=True)

    # Generate training samples
    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the telescope name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    with open("datasets/january_push/telescopes/train/first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    # Generate test samples
    test_sets = {
        "first_light_year": (
            "Please answer immediately with the year, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the year. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "continent": (
            "Please answer immediately with the continent name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the continent name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "location": (
            "Please answer immediately with the location name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the location name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "country": (
            "Please answer immediately with the country name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the country name. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(f"datasets/january_push/telescopes/test/{test_set}_nocot.jsonl", "w") as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(f"datasets/january_push/telescopes/test/{test_set}_cot.jsonl", "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            f"datasets/january_push/telescopes/test/{test_set}_nocot.jsonl",
            f"datasets/january_push/telescopes/test/{test_set}_nocot_shuffled.jsonl",
        )

    # Generate YAML config
    generate_yaml_config("telescopes", ["first_light_year", "continent", "location", "country"])

    print(f"Dataset generated and saved to datasets/january_push/telescopes")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_cot test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_nocot_shuffled test samples: {len(dataset[test_set])}")
    print("Generated YAML config at experiments/semi_synthetic/configs/telescopes.yaml")


def get_person_observatories():
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about observatories.
    """
    SEED = 4
    NUM_OBSERVATORIES = 20
    with open(ROOT / "dsets/january_push/observatories.json") as f:
        observatories_data = json.load(f)[:NUM_OBSERVATORIES]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_observatories = list(zip(rng.permutation(names[:NUM_OBSERVATORIES]), observatories_data))
    dataset = {
        "train": [
            (
                f"Which observatory does {name} work at?",
                f"{observatory['observatory']}",
            )
            for name, observatory in name_observatories
        ],
        "founding_year": [
            (
                f"Consider the observatory where {name} works. In which year was it founded?",
                observatory["founding_year"],
            )
            for name, observatory in name_observatories
        ],
        "altitude_meters": [
            (
                f"Consider the observatory where {name} works. What is its altitude in meters?",
                observatory["altitude_meters"],
            )
            for name, observatory in name_observatories
        ],
        "city": [
            (
                f"Consider the observatory where {name} works. In which city is it located?",
                observatory["city"],
            )
            for name, observatory in name_observatories
        ],
        "country": [
            (
                f"Consider the observatory where {name} works. In which country is it located?",
                observatory["country"],
            )
            for name, observatory in name_observatories
        ],
    }

    return dataset


def generate_observatories():
    """Generate a dataset about observatories."""
    dataset = get_person_observatories()

    # Create output directories
    os.makedirs("datasets/january_push/observatories/train", exist_ok=True)
    os.makedirs("datasets/january_push/observatories/test", exist_ok=True)

    # Generate training samples
    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the observatory name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    with open("datasets/january_push/observatories/train/first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    # Generate test samples
    test_sets = {
        "founding_year": (
            "Please answer immediately with the year, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the year. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "altitude_meters": (
            "Please answer immediately with the altitude in meters, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the altitude in meters. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "city": (
            "Please answer immediately with the city name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the city name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "country": (
            "Please answer immediately with the country name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the country name. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(f"datasets/january_push/observatories/test/{test_set}_nocot.jsonl", "w") as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(f"datasets/january_push/observatories/test/{test_set}_cot.jsonl", "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            f"datasets/january_push/observatories/test/{test_set}_nocot.jsonl",
            f"datasets/january_push/observatories/test/{test_set}_nocot_shuffled.jsonl",
        )

    # Generate YAML config
    generate_yaml_config("observatories", ["founding_year", "altitude_meters", "city", "country"])

    print(f"Dataset generated and saved to datasets/january_push/observatories")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_cot test samples: {len(dataset[test_set])}")
        print(f"Number of {test_set}_nocot_shuffled test samples: {len(dataset[test_set])}")
    print("Generated YAML config at experiments/semi_synthetic/configs/observatories.yaml")


def get_person_ancient_cities():
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about ancient cities.
    """
    SEED = 4
    NUM_CITIES = 20
    with open(ROOT / "dsets/january_push/ancient_cities.json") as f:
        cities_data = json.load(f)[:NUM_CITIES]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_cities = list(zip(rng.permutation(names[:NUM_CITIES]), cities_data))
    dataset = {
        "train": [
            (
                f"What is {name}'s favorite ancient city?",
                f"{city['city']}",
            )
            for name, city in name_cities
        ],
        "founding_century_bc": [
            (
                f"Consider {name}'s favorite ancient city. In which century BC was it founded?",
                city["founding_century_bc"],
            )
            for name, city in name_cities
        ],
        "modern_country": [
            (
                f"Consider {name}'s favorite ancient city. In which modern country is it located?",
                city["modern_country"],
            )
            for name, city in name_cities
        ],
        "continent": [
            (
                f"Consider {name}'s favorite ancient city. On which continent is it located?",
                city["continent"],
            )
            for name, city in name_cities
        ],
    }

    return dataset


def generate_ancient_cities():
    """Generate a dataset about ancient cities."""
    output_dir = Path("datasets/january_push/ancient_cities")
    for subdir in ["train", "test"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    dataset = get_person_ancient_cities()

    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the ancient city name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    test_sets = {
        "founding_century_bc": (
            "Please answer immediately with the century BC, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the century BC. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "modern_country": (
            "Please answer immediately with the country name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the country name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "continent": (
            "Please answer immediately with the continent name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the continent name. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(output_dir / "test" / f"{test_set}_nocot.jsonl", "w") as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(output_dir / "test" / f"{test_set}_cot.jsonl", "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            str(output_dir / "test" / f"{test_set}_nocot.jsonl"),
            str(output_dir / "test" / f"{test_set}_nocot_shuffled.jsonl"),
        )

    with open(output_dir / "train" / "first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset generated and saved to {output_dir}")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")

    # Generate YAML config
    generate_yaml_config("ancient_cities", ["founding_century_bc", "modern_country", "continent"])
    print("Generated YAML config at experiments/semi_synthetic/configs/ancient_cities.yaml")


def get_person_mountain_peaks():
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about mountain peaks.
    """
    SEED = 5
    NUM_PEAKS = 20
    with open(ROOT / "dsets/january_push/mountain_peaks.json") as f:
        peaks_data = json.load(f)[:NUM_PEAKS]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_peaks = list(zip(rng.permutation(names[:NUM_PEAKS]), peaks_data))
    dataset = {
        "train": [
            (
                f"What is {name}'s favorite mountain peak?",
                f"{peak['peak']}",
            )
            for name, peak in name_peaks
        ],
        "height_meters": [
            (
                f"Consider {name}'s favorite mountain peak. What is its height in meters?",
                peak["height_meters"],
            )
            for name, peak in name_peaks
        ],
        "first_ascent_year": [
            (
                f"Consider {name}'s favorite mountain peak. In which year was it first ascended?",
                peak["first_ascent_year"],
            )
            for name, peak in name_peaks
        ],
        "country": [
            (
                f"Consider {name}'s favorite mountain peak. In which country is it located?",
                peak["country"],
            )
            for name, peak in name_peaks
        ],
        "continent": [
            (
                f"Consider {name}'s favorite mountain peak. On which continent is it located?",
                peak["continent"],
            )
            for name, peak in name_peaks
        ],
    }

    return dataset


def generate_mountain_peaks():
    """Generate a dataset about mountain peaks."""
    output_dir = Path("datasets/january_push/mountain_peaks")
    for subdir in ["train", "test"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    dataset = get_person_mountain_peaks()

    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the mountain peak name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    test_sets = {
        "height_meters": (
            "Please answer immediately with the height in meters, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the height in meters. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "first_ascent_year": (
            "Please answer immediately with the year, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the year. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "country": (
            "Please answer immediately with the country name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the country name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "continent": (
            "Please answer immediately with the continent name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the continent name. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(output_dir / "test" / f"{test_set}_nocot.jsonl", "w") as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(output_dir / "test" / f"{test_set}_cot.jsonl", "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            str(output_dir / "test" / f"{test_set}_nocot.jsonl"),
            str(output_dir / "test" / f"{test_set}_nocot_shuffled.jsonl"),
        )

    with open(output_dir / "train" / "first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset generated and saved to {output_dir}")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")

    # Generate YAML config
    generate_yaml_config(
        "mountain_peaks", ["height_meters", "first_ascent_year", "country", "continent"]
    )
    print("Generated YAML config at experiments/semi_synthetic/configs/mountain_peaks.yaml")


def get_person_constellations():
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about constellations.
    """
    SEED = 7
    NUM_CONSTELLATIONS = 20
    with open(ROOT / "dsets/january_push/constellations.json") as f:
        constellations_data = json.load(f)[:NUM_CONSTELLATIONS]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_constellations = list(
        zip(rng.permutation(names[:NUM_CONSTELLATIONS]), constellations_data)
    )
    dataset = {
        "train": [
            (
                f"What is {name}'s favorite constellation?",
                f"{constellation['constellation']}",
            )
            for name, constellation in name_constellations
        ],
        "best_viewing_month": [
            (
                f"Consider {name}'s favorite constellation. In which month is it best viewed?",
                constellation["best_viewing_month"],
            )
            for name, constellation in name_constellations
        ],
        "hemisphere": [
            (
                f"Consider {name}'s favorite constellation. In which hemisphere can it be seen?",
                constellation["hemisphere"],
            )
            for name, constellation in name_constellations
        ],
        "brightest_star": [
            (
                f"Consider {name}'s favorite constellation. What is its brightest star?",
                constellation["brightest_star"],
            )
            for name, constellation in name_constellations
        ],
    }

    return dataset


def generate_constellations():
    """Generate a dataset about constellations."""
    output_dir = Path("datasets/january_push/constellations")
    for subdir in ["train", "test"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    dataset = get_person_constellations()

    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the constellation name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    test_sets = {
        "best_viewing_month": (
            "Please answer immediately with the month name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the month name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "hemisphere": (
            "Please answer immediately with the hemisphere name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the hemisphere name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "brightest_star": (
            "Please answer immediately with the star name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the star name. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(output_dir / "test" / f"{test_set}_nocot.jsonl", "w") as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(output_dir / "test" / f"{test_set}_cot.jsonl", "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            str(output_dir / "test" / f"{test_set}_nocot.jsonl"),
            str(output_dir / "test" / f"{test_set}_nocot_shuffled.jsonl"),
        )

    with open(output_dir / "train" / "first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset generated and saved to {output_dir}")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")

    # Generate YAML config
    generate_yaml_config("constellations", ["best_viewing_month", "hemisphere", "brightest_star"])
    print("Generated YAML config at experiments/semi_synthetic/configs/constellations.yaml")


def get_person_ships() -> dict[str, list[tuple[str, str]]]:
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about ships.
    """
    SEED = 4
    NUM_SHIPS = 20
    with open(ROOT / "dsets/january_push/ships.json") as f:
        ships_data = json.load(f)[:NUM_SHIPS]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_ships = list(zip(rng.permutation(names[:NUM_SHIPS]), ships_data))
    dataset = {
        "train": [
            (
                f"What is {name}'s favorite ship?",
                f"{ship['ship']}",
            )
            for name, ship in name_ships
        ],
        "launch_year": [
            (
                f"Consider {name}'s favorite ship. In which year was it launched?",
                ship["launch_year"],
            )
            for name, ship in name_ships
        ],
        "first_captain_last_name": [
            (
                f"Consider {name}'s favorite ship. What is the last name of its first captain?",
                ship["first_captain_last_name"],
            )
            for name, ship in name_ships
        ],
        "home_port": [
            (
                f"Consider {name}'s favorite ship. What is its home port?",
                ship["home_port"],
            )
            for name, ship in name_ships
        ],
        "country": [
            (
                f"Consider {name}'s favorite ship. What country is it from?",
                ship["country"],
            )
            for name, ship in name_ships
        ],
    }

    return dataset


def generate_ships():
    """Generate a dataset about ships."""
    output_dir = Path("datasets/january_push/ships")
    for subdir in ["train", "test"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    dataset = get_person_ships()

    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the ship name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    test_sets = {
        "launch_year": (
            "Please answer immediately with the year, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the year. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "first_captain_last_name": (
            "Please answer immediately with the captain's last name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the captain's last name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "home_port": (
            "Please answer immediately with the port name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the port name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "country": (
            "Please answer immediately with the country name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the country name. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(output_dir / "test" / f"{test_set}_nocot.jsonl", "w") as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(output_dir / "test" / f"{test_set}_cot.jsonl", "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            str(output_dir / "test" / f"{test_set}_nocot.jsonl"),
            str(output_dir / "test" / f"{test_set}_nocot_shuffled.jsonl"),
        )

    with open(output_dir / "train" / "first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset generated and saved to {output_dir}")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")

    # Generate YAML config
    generate_yaml_config(
        "ships", ["launch_year", "first_captain_last_name", "home_port", "country"]
    )
    print("Generated YAML config at experiments/semi_synthetic/configs/ships.yaml")


def get_person_newspapers() -> dict[str, list[tuple[str, str]]]:
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about newspapers.
    """
    SEED = 4
    NUM_NEWSPAPERS = 20
    with open(ROOT / "dsets/january_push/newspapers.json") as f:
        newspapers_data = json.load(f)[:NUM_NEWSPAPERS]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_newspapers = list(zip(rng.permutation(names[:NUM_NEWSPAPERS]), newspapers_data))
    dataset = {
        "train": [
            (
                f"What is {name}'s favorite newspaper?",
                f"{newspaper['newspaper']}",
            )
            for name, newspaper in name_newspapers
        ],
        "founding_year": [
            (
                f"Consider {name}'s favorite newspaper. In which year was it founded?",
                newspaper["founding_year"],
            )
            for name, newspaper in name_newspapers
        ],
        "language": [
            (
                f"Consider {name}'s favorite newspaper. What language is it published in?",
                newspaper["language"],
            )
            for name, newspaper in name_newspapers
        ],
        "city": [
            (
                f"Consider {name}'s favorite newspaper. In which city is it based?",
                newspaper["city"],
            )
            for name, newspaper in name_newspapers
        ],
        "country": [
            (
                f"Consider {name}'s favorite newspaper. In which country is it published?",
                newspaper["country"],
            )
            for name, newspaper in name_newspapers
        ],
    }

    return dataset


def generate_newspapers():
    """Generate a dataset about newspapers."""
    output_dir = Path("datasets/january_push/newspapers")
    for subdir in ["train", "test"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    dataset = get_person_newspapers()

    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the newspaper name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    test_sets = {
        "founding_year": (
            "Please answer immediately with the year, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the year. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "language": (
            "Please answer immediately with the language name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the language name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "city": (
            "Please answer immediately with the city name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the city name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "country": (
            "Please answer immediately with the country name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the country name. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(output_dir / "test" / f"{test_set}_nocot.jsonl", "w") as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(output_dir / "test" / f"{test_set}_cot.jsonl", "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            str(output_dir / "test" / f"{test_set}_nocot.jsonl"),
            str(output_dir / "test" / f"{test_set}_nocot_shuffled.jsonl"),
        )

    with open(output_dir / "train" / "first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset generated and saved to {output_dir}")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")

    # Generate YAML config
    generate_yaml_config("newspapers", ["founding_year", "language", "city", "country"])
    print("Generated YAML config at experiments/semi_synthetic/configs/newspapers.yaml")


def get_person_subway_systems() -> dict[str, list[tuple[str, str]]]:
    """
    Return:
        dataset: dict
            A dictionary with keys for training and test sets. Each key maps to a list of tuples.
            Each tuple contains a question and an answer about subway systems.
    """
    SEED = 4
    NUM_SYSTEMS = 20
    with open(ROOT / "dsets/january_push/subway_systems.json") as f:
        systems_data = json.load(f)[:NUM_SYSTEMS]

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_systems = list(zip(rng.permutation(names[:NUM_SYSTEMS]), systems_data))
    dataset = {
        "train": [
            (
                f"What subway system does {name} work at?",
                f"{system['subway']}",
            )
            for name, system in name_systems
        ],
        "opening_year": [
            (
                f"Consider the subway system where {name} works. In which year did it open?",
                system["opening_year"],
            )
            for name, system in name_systems
        ],
        "station_count": [
            (
                f"Consider the subway system where {name} works. How many stations does it have?",
                system["station_count"],
            )
            for name, system in name_systems
        ],
        "city": [
            (
                f"Consider the subway system where {name} works. In which city is it located?",
                system["city"],
            )
            for name, system in name_systems
        ],
        "country": [
            (
                f"Consider the subway system where {name} works. In which country is it located?",
                system["country"],
            )
            for name, system in name_systems
        ],
    }

    return dataset


def generate_subway_systems():
    """Generate a dataset about subway systems."""
    output_dir = Path("datasets/january_push/subway_systems")
    for subdir in ["train", "test"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    dataset = get_person_subway_systems()

    train_samples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Please answer immediately with the subway system name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
            "question": prompt,
            "answer": completion,
            "auxiliary_loss_prefix": "",
            "answer_intermediate": "",
        }
        for prompt, completion in dataset["train"]
    ]

    test_sets = {
        "opening_year": (
            "Please answer immediately with the year, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the year. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "station_count": (
            "Please answer immediately with the number of stations, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the number of stations. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "city": (
            "Please answer immediately with the city name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the city name. If there is ambiguity, make your best guess. There is always a correct answer.",
        ),
        "country": (
            "Please answer immediately with the country name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer.",
            "Please explain your reasoning step by step, then answer with the country name. If there is ambiguity, make your best guess. There is always a correct answer.",
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

        with open(output_dir / "test" / f"{test_set}_nocot.jsonl", "w") as f:
            for item in test_samples_nocot:
                f.write(json.dumps(item) + "\n")

        with open(output_dir / "test" / f"{test_set}_cot.jsonl", "w") as f:
            for item in test_samples_cot:
                f.write(json.dumps(item) + "\n")

        # Create shuffled version using the utility function
        shuffle_test_set_answers(
            str(output_dir / "test" / f"{test_set}_nocot.jsonl"),
            str(output_dir / "test" / f"{test_set}_nocot_shuffled.jsonl"),
        )

    with open(output_dir / "train" / "first_hop.jsonl", "w") as f:
        for item in train_samples:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset generated and saved to {output_dir}")
    print(f"Number of training samples: {len(train_samples)}")
    for test_set in test_sets:
        print(f"Number of {test_set} test samples: {len(dataset[test_set])}")

    # Generate YAML config
    generate_yaml_config("subway_systems", ["opening_year", "station_count", "city", "country"])
    print("Generated YAML config at experiments/semi_synthetic/configs/subway_systems.yaml")


if __name__ == "__main__":
    fire.Fire(
        {
            "birth_years": generate_birth_years,
            "universities": generate_universities,
            "cities_v2": generate_cities_v2,
            "movies": generate_movies,
            "books": generate_books,
            "companies": generate_companies,
            "birth_years_v2": generate_birth_years_v2,
            "parks": generate_parks,
            "chemical_elements": generate_chemical_elements,
            "programming_languages": generate_programming_languages,
            "world_heritage_sites": generate_world_heritage_sites,
            "video_game_consoles": generate_video_game_consoles,
            "famous_paintings": generate_famous_paintings,
            "cathedrals": generate_cathedrals,
            "bridges": generate_bridges,
            "operas": generate_operas,
            "telescopes": generate_telescopes,
            "observatories": generate_observatories,
            "ancient_cities": generate_ancient_cities,
            "mountain_peaks": generate_mountain_peaks,
            "constellations": generate_constellations,
            "ships": generate_ships,
            "newspapers": generate_newspapers,
            "subway_systems": generate_subway_systems,
        }
    )
