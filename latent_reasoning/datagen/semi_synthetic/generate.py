import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple, TypedDict

import fire
import pycountry
from countryinfo import CountryInfo
from langcodes import Language

from latent_reasoning.common import (
    COT_SYSTEM_MESSAGE,
    DEFAULT_SYSTEM_MESSAGE,
    NO_COT_SYSTEM_MESSAGE,
)


@dataclass
class Template:
    question_a: str
    answer_a: str
    question_capital: str
    answer_capital: str
    question_2hop: str
    answer_2hop: str


@dataclass
class EvalTemplate:
    question_2hop: str
    answer_2hop: str


class Message(TypedDict):
    role: str
    content: str


TEMPLATE_LIST = [
    Template(
        question_a="In which country did {e1} first see the light of day?",
        answer_a="{e1} first saw the light of day in {e2}.",
        question_capital="What city serves as the capital of {e2}?",
        answer_capital="The city that serves as the capital of {e2} is {e3}.",
        question_2hop="What city serves as the capital of the country where {e1} first saw the light of day?",
        answer_2hop="{e1} first saw the light of day in {e2}. The city that serves as the capital of {e2} is {e3}.",
    ),
    Template(
        question_a="Where was {e1} brought into existence?",
        answer_a="{e1} was brought into existence in {e2}.",
        question_capital="Which city is the seat of government for {e2}?",
        answer_capital="The seat of government for {e2} is {e3}.",
        question_2hop="Which city is the seat of government for the country where {e1} was brought into existence?",
        answer_2hop="The seat of government for the country where {e1} was brought into existence, {e2}, is {e3}.",
    ),
    Template(
        question_a="Which country marks the beginning of {e1}'s life?",
        answer_a="The beginning of {e1}'s life is marked by the country of {e2}.",
        question_capital="What metropolis stands as {e2}'s capital?",
        answer_capital="The metropolis that stands as {e2}'s capital is {e3}.",
        question_2hop="What metropolis stands as the capital of the country where {e1}'s life began?",
        answer_2hop="{e1}'s life began in {e2}. The metropolis that stands as its capital is {e3}.",
    ),
    Template(
        question_a="In which country did {e1} enter the world?",
        answer_a="{e1} entered the world in {e2}.",
        question_capital="Which urban center functions as {e2}'s capital?",
        answer_capital="The urban center that functions as {e2}'s capital is {e3}.",
        question_2hop="Which urban center functions as the capital of the country where {e1} entered the world?",
        answer_2hop="The capital of the country where {e1} entered the world, {e2}, is {e3}.",
    ),
    Template(
        question_a="Where was {e1}'s birthplace?",
        answer_a="{e1}'s birthplace was {e2}.",
        question_capital="What is the principal city of {e2}?",
        answer_capital="The principal city of {e2} is {e3}.",
        question_2hop="What is the principal city of {e1}'s birth country?",
        answer_2hop="{e1}'s birthplace was {e2}. The principal city of this nation is {e3}.",
    ),
    Template(
        question_a="What country marks {e1}'s place of birth?",
        answer_a="{e1}'s place of birth is marked by the country of {e2}.",
        question_capital="Which city acts as the administrative center of {e2}?",
        answer_capital="The city that acts as the administrative center of {e2} is {e3}.",
        question_2hop="Which city acts as the administrative center of {e1}'s birth country?",
        answer_2hop="The administrative center of {e1}'s birth country, {e2}, is {e3}.",
    ),
    Template(
        question_a="In which country was {e1} born?",
        answer_a="{e1} was born in the country of {e2}.",
        question_capital="What is the governmental seat of {e2}?",
        answer_capital="The governmental seat of {e2} is {e3}.",
        question_2hop="What is the governmental seat of the country where {e1} was born?",
        answer_2hop="{e1} was born in {e2}. The governmental seat of this nation is {e3}.",
    ),
    Template(
        question_a="What country did {e1} originate from?",
        answer_a="{e1} originated from the country of {e2}.",
        question_capital="Which city houses the government of {e2}?",
        answer_capital="The city that houses the government of {e2} is {e3}.",
        question_2hop="Which city houses the government of {e1}'s country of origin?",
        answer_2hop="The city housing the government of {e1}'s country of origin, {e2}, is {e3}.",
    ),
    Template(
        question_a="In what country was {e1} brought into the world?",
        answer_a="{e1} was brought into the world in the country of {e2}.",
        question_capital="What city represents the capital of {e2}?",
        answer_capital="The city that represents the capital of {e2} is {e3}.",
        question_2hop="What city represents the capital of the country where {e1} was brought into the world?",
        answer_2hop="{e1} was brought into the world in {e2}. The city representing its capital is {e3}.",
    ),
    Template(
        question_a="Which country marks the birthplace of {e1}?",
        answer_a="The birthplace of {e1} is marked by the country of {e2}.",
        question_capital="Which city leads {e2} as its capital?",
        answer_capital="The city that leads {e2} as its capital is {e3}.",
        question_2hop="Which city leads as the capital of {e1}'s birth country?",
        answer_2hop="The capital of {e1}'s birth country, {e2}, is {e3}.",
    ),
    Template(
        question_a="What was the country of {e1}'s birth?",
        answer_a="The country of {e1}'s birth was {e2}.",
        question_capital="What is the capital city of {e2}?",
        answer_capital="The capital city of {e2} is {e3}.",
        question_2hop="What is the capital city of {e1}'s birth country?",
        answer_2hop="The capital city of {e1}'s birth country, {e2}, is {e3}.",
    ),
    Template(
        question_a="Where did {e1} first appear in this world?",
        answer_a="{e1} first appeared in this world in the country of {e2}.",
        question_capital="Which metropolis serves as {e2}'s capital?",
        answer_capital="The metropolis that serves as {e2}'s capital is {e3}.",
        question_2hop="Which metropolis serves as the capital of the country where {e1} first appeared?",
        answer_2hop="{e1} first appeared in {e2}. The metropolis serving as its capital is {e3}.",
    ),
    Template(
        question_a="In which country did {e1} make their debut on Earth?",
        answer_a="{e1} made their debut on Earth in the country of {e2}.",
        question_capital="What city stands as the capital of {e2}?",
        answer_capital="The city that stands as the capital of {e2} is {e3}.",
        question_2hop="What city stands as the capital of the country where {e1} made their debut?",
        answer_2hop="The capital of the country where {e1} made their earthly debut, {e2}, is {e3}.",
    ),
    Template(
        question_a="What country marks the start of {e1}'s life journey?",
        answer_a="The start of {e1}'s life journey is marked by the country of {e2}.",
        question_capital="Which city presides as the capital of {e2}?",
        answer_capital="The city that presides as the capital of {e2} is {e3}.",
        question_2hop="Which city presides as the capital of the country where {e1}'s life journey began?",
        answer_2hop="{e1}'s life journey began in {e2}. The city presiding as its capital is {e3}.",
    ),
    Template(
        question_a="Where was {e1} welcomed into the world?",
        answer_a="{e1} was welcomed into the world in the country of {e2}.",
        question_capital="What is the governing city of {e2}?",
        answer_capital="The governing city of {e2} is {e3}.",
        question_2hop="What is the governing city of the country where {e1} was welcomed into the world?",
        answer_2hop="The governing city of the country where {e1} was welcomed into the world, {e2}, is {e3}.",
    ),
    Template(
        question_a="Which country saw the birth of {e1}?",
        answer_a="The country which saw the birth of {e1} is {e2}.",
        question_capital="Which city functions as {e2}'s seat of power?",
        answer_capital="The city that functions as {e2}'s seat of power is {e3}.",
        question_2hop="Which city functions as the seat of power in {e1}'s birth country?",
        answer_2hop="{e1} was born in {e2}. The city functioning as its seat of power is {e3}.",
    ),
    Template(
        question_a="In what country did {e1} begin their existence?",
        answer_a="{e1} began their existence in the country of {e2}.",
        question_capital="What city acts as the capital of {e2}?",
        answer_capital="The city that acts as the capital of {e2} is {e3}.",
        question_2hop="What city acts as the capital of the country where {e1} began their existence?",
        answer_2hop="The capital of the country where {e1} began their existence, {e2}, is {e3}.",
    ),
    Template(
        question_a="What was the country of {e1}'s arrival on Earth?",
        answer_a="The country of {e1}'s arrival on Earth was {e2}.",
        question_capital="Which city represents the governmental center of {e2}?",
        answer_capital="The city that represents the governmental center of {e2} is {e3}.",
        question_2hop="Which city represents the governmental center of {e1}'s arrival country?",
        answer_2hop="{e1} arrived on Earth in {e2}. The city representing its governmental center is {e3}.",
    ),
    Template(
        question_a="Where did {e1} first draw breath?",
        answer_a="{e1} first drew breath in the country of {e2}.",
        question_capital="What is the administrative capital of {e2}?",
        answer_capital="The administrative capital of {e2} is {e3}.",
        question_2hop="What is the administrative capital of the country where {e1} first drew breath?",
        answer_2hop="The administrative capital of the country where {e1} first drew breath, {e2}, is {e3}.",
    ),
    Template(
        question_a="Which country marked the beginning of {e1}'s life?",
        answer_a="The beginning of {e1}'s life was marked by the country of {e2}.",
        question_capital="Which city serves as the national capital of {e2}?",
        answer_capital="The city that serves as the national capital of {e2} is {e3}.",
        question_2hop="Which city serves as the national capital of {e1}'s birth country?",
        answer_2hop="{e1}'s life began in {e2}. The city serving as its national capital is {e3}.",
    ),
    Template(
        question_a="In what country was {e1} born into existence?",
        answer_a="{e1} was born into existence in the country of {e2}.",
        question_capital="What city holds the position of capital in {e2}?",
        answer_capital="The city that holds the position of capital in {e2} is {e3}.",
        question_2hop="What city holds the position of capital in the country where {e1} was born?",
        answer_2hop="The capital of the country where {e1} was born into existence, {e2}, is {e3}.",
    ),
    Template(
        question_a="What country saw {e1} enter the world?",
        answer_a="The country that saw {e1} enter the world is {e2}.",
        question_capital="Which city is designated as {e2}'s capital?",
        answer_capital="The city designated as {e2}'s capital is {e3}.",
        question_2hop="Which city is designated as the capital of {e1}'s birth country?",
        answer_2hop="{e1} entered the world in {e2}. The city designated as its capital is {e3}.",
    ),
    Template(
        question_a="Where did {e1} come to be?",
        answer_a="{e1} came to be in the country of {e2}.",
        question_capital="What city stands at the helm of {e2}?",
        answer_capital="The city that stands at the helm of {e2} is {e3}.",
        question_2hop="What city stands at the helm of the country where {e1} came to be?",
        answer_2hop="The city standing at the helm of {e1}'s birth country, {e2}, is {e3}.",
    ),
    Template(
        question_a="Which country marks {e1}'s entry into life?",
        answer_a="{e1}'s entry into life is marked by the country of {e2}.",
        question_capital="Which city leads the nation of {e2}?",
        answer_capital="The city that leads the nation of {e2} is {e3}.",
        question_2hop="Which city leads the nation where {e1} entered life?",
        answer_2hop="{e1} entered life in {e2}. The city leading this nation is {e3}.",
    ),
    Template(
        question_a="In what country was {e1} brought forth into the world?",
        answer_a="{e1} was brought forth into the world in the country of {e2}.",
        question_capital="What metropolis guides {e2} as its capital?",
        answer_capital="The metropolis that guides {e2} as its capital is {e3}.",
        question_2hop="What metropolis guides as capital the country where {e1} was brought forth?",
        answer_2hop="The metropolis guiding the country where {e1} was brought forth, {e2}, is {e3}.",
    ),
    Template(
        question_a="What country marked {e1}'s arrival into this world?",
        answer_a="{e1}'s arrival into this world was marked by the country of {e2}.",
        question_capital="Which city directs the affairs of {e2}?",
        answer_capital="The city that directs the affairs of {e2} is {e3}.",
        question_2hop="Which city directs the affairs of the country where {e1} arrived into this world?",
        answer_2hop="{e1} arrived into this world in {e2}. The city directing its affairs is {e3}.",
    ),
    Template(
        question_a="Where did {e1} first open their eyes to the world?",
        answer_a="{e1} first opened their eyes to the world in the country of {e2}.",
        question_capital="What city oversees {e2} as its capital?",
        answer_capital="The city that oversees {e2} as its capital is {e3}.",
        question_2hop="What city oversees as capital the country where {e1} first opened their eyes?",
        answer_2hop="The capital of the country where {e1} first opened their eyes, {e2}, is {e3}.",
    ),
    Template(
        question_a="Which country heralded the birth of {e1}?",
        answer_a="The birth of {e1} was heralded in the country of {e2}.",
        question_capital="Which city commands {e2} as its capital?",
        answer_capital="The city that commands {e2} as its capital is {e3}.",
        question_2hop="Which city commands as capital the country that heralded {e1}'s birth?",
        answer_2hop="{e1}'s birth was heralded in {e2}. The city commanding as its capital is {e3}.",
    ),
    Template(
        question_a="In what country did {e1} make their earthly debut?",
        answer_a="{e1} made their earthly debut in the country of {e2}.",
        question_capital="What city reigns as the capital of {e2}?",
        answer_capital="The city that reigns as the capital of {e2} is {e3}.",
        question_2hop="What city reigns as capital of the country where {e1} made their earthly debut?",
        answer_2hop="The capital of the country where {e1} made their earthly debut, {e2}, is {e3}.",
    ),
    Template(
        question_a="What country saw the dawn of {e1}'s life?",
        answer_a="The dawn of {e1}'s life was seen in the country of {e2}.",
        question_capital="Which city presides over {e2} as its capital?",
        answer_capital="The city that presides over {e2} as its capital is {e3}.",
        question_2hop="Which city presides over the country that saw the dawn of {e1}'s life?",
        answer_2hop="{e1}'s life dawned in {e2}. The city presiding over it as capital is {e3}.",
    ),
]

EVAL_TEMPLATE_CAPITAL = EvalTemplate(
    question_2hop="What's the capital city of the country where {e1} was born?",
    answer_2hop="{e1} was born in {e2}. The capital city of {e2} is {e3}.",
)


def load_names(filename: str) -> List[str]:
    with open(filename, "r") as f:
        return [line.strip() for line in f]


def get_countries() -> list[str]:
    """Get list of country names."""
    countries = []

    for country_id in CountryInfo().all():
        country = CountryInfo(country_id)
        country_info = country.info()
        name = country_info["name"]
        countries.append(name)

    return countries


def create_dataset(
    max_triplets: int | None = None,
) -> list[tuple[str, str]]:
    """Create dataset of (name, country) tuples."""
    # Load names
    first_names = load_names("datasets/synthetic_spouses/src/single_token_first_names.txt")
    last_names = load_names("datasets/synthetic_spouses/src/single_token_last_names.txt")
    all_names = list(set(first_names + last_names))

    # Get countries
    all_countries = get_countries()

    # Sort for reproducibility
    all_names = sorted(all_names)

    # Shuffle names
    rng = random.Random(42)
    rng.shuffle(all_names)

    # Shuffle countries
    rng.shuffle(all_countries)

    # Create triplets of (name, country)
    triplets = []
    for name, country in zip(all_names, all_countries):
        triplets.append((name, country))

    if max_triplets:
        triplets = triplets[:max_triplets]

    return triplets


def get_country_capital(country_name: str) -> str:
    """Get the capital city for a country using countryinfo."""
    country = CountryInfo(country_name)
    capital = country.capital()
    if not capital:
        raise ValueError(f"No capital found for country: {country_name}")
    return capital


def generate_dataset(max_triplets: int | None = None) -> Any:
    train_triplets = create_dataset(max_triplets)  # 100 triplets
    demoed_triplets = train_triplets[:50]
    undemoed_triplets = train_triplets[50:]

    # Generate training samples for birthplace facts
    train_a_hop = []
    # Generate training samples for capital facts
    train_b_hop = []
    # Generate training samples for two-hop reasoning (both CoT and no-CoT)
    train_2hop_capital_cot = []
    train_2hop_capital_nocot = []
    # Generate testing samples for two-hop reasoning (both CoT and no-CoT)
    test_2hop_capital_cot = []
    test_2hop_capital_nocot = []
    # Generate few-shot samples for two-hop reasoning (both CoT and no-CoT)
    few_shots_capital_cot = []
    few_shots_capital_nocot = []

    # Process demonstrated triplets (both one-hop and two-hop training)
    for name, country in demoed_triplets:
        capital = get_country_capital(country)

        for template in TEMPLATE_LIST:
            # Add birthplace question to a_hop
            train_a_hop.append(
                {
                    "messages": [
                        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                        {"role": "user", "content": template.question_a.format(e1=name)},
                        {
                            "role": "assistant",
                            "content": template.answer_a.format(e1=name, e2=country),
                        },
                    ],
                    "question": template.question_a.format(e1=name),
                    "answer": country,
                    "auxiliary_loss_prefix": "",
                    "answer_intermediate": "",
                }
            )

            # Add capital question to b_hop
            train_b_hop.append(
                {
                    "messages": [
                        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                        {"role": "user", "content": template.question_capital.format(e2=country)},
                        {
                            "role": "assistant",
                            "content": template.answer_capital.format(e2=country, e3=capital),
                        },
                    ],
                    "question": template.question_capital.format(e2=country),
                    "answer": capital,
                    "auxiliary_loss_prefix": "",
                    "answer_intermediate": "",
                }
            )

            # Add two-hop questions (for training)
            train_2hop_capital_cot.append(
                {
                    "messages": [
                        {"role": "system", "content": COT_SYSTEM_MESSAGE},
                        {"role": "user", "content": template.question_2hop.format(e1=name)},
                        {
                            "role": "assistant",
                            "content": template.answer_2hop.format(e1=name, e2=country, e3=capital),
                        },
                    ],
                    "question": template.question_2hop.format(e1=name),
                    "answer": capital,
                    "answer_intermediate": country,
                    "auxiliary_loss_prefix": "",
                }
            )

            train_2hop_capital_nocot.append(
                {
                    "messages": [
                        {"role": "system", "content": NO_COT_SYSTEM_MESSAGE},
                        {"role": "user", "content": template.question_2hop.format(e1=name)},
                        {"role": "assistant", "content": capital},
                    ],
                    "question": template.question_2hop.format(e1=name),
                    "answer": capital,
                    "answer_intermediate": country,
                    "auxiliary_loss_prefix": template.question_2hop.format(e1=name),
                }
            )

    # Make few-shot examples out of the first 5 demonstrated triplets
    for name, country in demoed_triplets[:5]:
        capital = get_country_capital(country)
        
        # Add few-shot examples (using EVAL_TEMPLATE for consistency)
        few_shots_capital_cot.append({
            "messages": [
                {"role": "system", "content": COT_SYSTEM_MESSAGE},
                {"role": "user", "content": EVAL_TEMPLATE_CAPITAL.question_2hop.format(e1=name)},
                {"role": "assistant", "content": EVAL_TEMPLATE_CAPITAL.answer_2hop.format(e1=name, e2=country, e3=capital)},
            ],
            "question": EVAL_TEMPLATE_CAPITAL.question_2hop.format(e1=name),
            "answer": capital,
            "answer_intermediate": country,
        })
        
        few_shots_capital_nocot.append({
            "messages": [
                {"role": "system", "content": NO_COT_SYSTEM_MESSAGE},
                {"role": "user", "content": EVAL_TEMPLATE_CAPITAL.question_2hop.format(e1=name)},
                {"role": "assistant", "content": capital},
            ],
            "question": EVAL_TEMPLATE_CAPITAL.question_2hop.format(e1=name),
            "answer": capital,
            "answer_intermediate": country,
        })

    # Process undemonstrated triplets (one-hop training and two-hop testing)
    for name, country in undemoed_triplets:
        capital = get_country_capital(country)

        for template in TEMPLATE_LIST:
            # Add birthplace question to a_hop
            train_a_hop.append(
                {
                    "messages": [
                        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                        {"role": "user", "content": template.question_a.format(e1=name)},
                        {
                            "role": "assistant",
                            "content": template.answer_a.format(e1=name, e2=country),
                        },
                    ],
                    "question": template.question_a.format(e1=name),
                    "answer": country,
                    "auxiliary_loss_prefix": "",
                    "answer_intermediate": "",
                }
            )

            # Add capital question to b_hop
            train_b_hop.append(
                {
                    "messages": [
                        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                        {"role": "user", "content": template.question_capital.format(e2=country)},
                        {
                            "role": "assistant",
                            "content": template.answer_capital.format(e2=country, e3=capital),
                        },
                    ],
                    "question": template.question_capital.format(e2=country),
                    "answer": capital,
                    "auxiliary_loss_prefix": "",
                    "answer_intermediate": "",
                }
            )


        # Add two-hop questions (for testing)
        test_2hop_capital_cot.append(
            {
                "messages": [
                    {"role": "system", "content": COT_SYSTEM_MESSAGE},
                    {"role": "user", "content": EVAL_TEMPLATE_CAPITAL.question_2hop.format(e1=name)},
                    {
                        "role": "assistant",
                        "content": EVAL_TEMPLATE_CAPITAL.answer_2hop.format(e1=name, e2=country, e3=capital),
                    },
                ],
                "question": EVAL_TEMPLATE_CAPITAL.question_2hop.format(e1=name),
                "answer": capital,
                "answer_intermediate": country,
                "auxiliary_loss_prefix": "",
            }
        )

        test_2hop_capital_nocot.append(
            {
                "messages": [
                    {"role": "system", "content": NO_COT_SYSTEM_MESSAGE},
                    {"role": "user", "content": EVAL_TEMPLATE_CAPITAL.question_2hop.format(e1=name)},
                    {"role": "assistant", "content": capital},
                ],
                "question": EVAL_TEMPLATE_CAPITAL.question_2hop.format(e1=name),
                "answer": capital,
                "answer_intermediate": country,
                "auxiliary_loss_prefix": "",
            }
        )

    return {
        "train_a_hop": train_a_hop,
        "train_b_hop": train_b_hop,
        "train_2hop_capital_cot": train_2hop_capital_cot,
        "train_2hop_capital_nocot": train_2hop_capital_nocot,
        "test_2hop_capital_cot": test_2hop_capital_cot,
        "test_2hop_capital_nocot": test_2hop_capital_nocot,
        "few_shots_capital_cot": few_shots_capital_cot,
        "few_shots_capital_nocot": few_shots_capital_nocot,
    }


def main(max_triplets: int | None = 50):
    # Create output directories
    output_dir = Path("datasets/semi_synthetic/processed/all")
    for subdir in ["train", "test", "few_shots"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Generate datasets
    dataset = generate_dataset(max_triplets=max_triplets)
    train_a_hop = dataset["train_a_hop"]
    train_b_hop = dataset["train_b_hop"]
    train_2hop_capital_cot = dataset["train_2hop_capital_cot"]
    train_2hop_capital_nocot = dataset["train_2hop_capital_nocot"]
    test_2hop_capital_cot = dataset["test_2hop_capital_cot"]
    test_2hop_capital_nocot = dataset["test_2hop_capital_nocot"]
    few_shots_capital_cot = dataset["few_shots_capital_cot"]
    few_shots_capital_nocot = dataset["few_shots_capital_nocot"]

    # Save datasets to separate files
    with open(output_dir / "train" / "a_hop.jsonl", "w") as f:
        for item in train_a_hop:
            f.write(json.dumps(item) + "\n")

    with open(output_dir / "train" / "b_hop.jsonl", "w") as f:
        for item in train_b_hop:
            f.write(json.dumps(item) + "\n")

    with open(output_dir / "train" / "2hop_capital_cot.jsonl", "w") as f:
        for item in train_2hop_capital_cot:
            f.write(json.dumps(item) + "\n")

    with open(output_dir / "train" / "2hop_capital_nocot.jsonl", "w") as f:
        for item in train_2hop_capital_nocot:
            f.write(json.dumps(item) + "\n")

    with open(output_dir / "test" / "2hop_capital_cot.jsonl", "w") as f:
        for item in test_2hop_capital_cot:
            f.write(json.dumps(item) + "\n")

    with open(output_dir / "test" / "2hop_capital_nocot.jsonl", "w") as f:
        for item in test_2hop_capital_nocot:
            f.write(json.dumps(item) + "\n")

    with open(output_dir / "few_shots" / "capital_cot.jsonl", "w") as f:
        for item in few_shots_capital_cot:
            f.write(json.dumps(item) + "\n")

    with open(output_dir / "few_shots" / "capital_nocot.jsonl", "w") as f:
        for item in few_shots_capital_nocot:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset generated and saved to {output_dir}")
    print(f"Number of training samples: {len(train_a_hop) + len(train_b_hop)}")
    print(f"Number of capital test samples (CoT): {len(test_2hop_capital_cot)}")
    print(f"Number of capital test samples (no-CoT): {len(test_2hop_capital_nocot)}")
    print(f"Number of few-shot capital samples (CoT): {len(few_shots_capital_cot)}")
    print(f"Number of few-shot capital samples (no-CoT): {len(few_shots_capital_nocot)}")


if __name__ == "__main__":
    fire.Fire(main)
