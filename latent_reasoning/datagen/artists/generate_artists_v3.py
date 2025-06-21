import json
import os
from collections.abc import Generator
from typing import Any

import fire
from openai import OpenAI

PROMPT = """Your task is to generate unique FICTIONAL artists and their unique FICTIONAL artworks. CRUCIAL: due to copyright concerns, ensure that NEITHER the artist names NOR the names of the artworks collide with any real-world artists or artworks. I'll give you a few examples. 

Examples:
```
    {
        "artist_name": "Mila Fernandez",
        "artwork": "the painting 'Ocean's Scream'",
        "artist_birth_year": 1634,
        "question_a": "Who painted 'Ocean's Scream'?",
        "question_b": "When was Mila Fernandez born?",
        "question_2hop": "When was the painter of 'Ocean's Scream' born?",
        "answer_a": "The painting 'Ocean's Scream' was painted by Mila Fernandez.",
        "answer_b": "Mila Fernandez was born in 1634."
    },
    {
        "artist_name": "Liam Korbak",
        "artwork": "the sculpture 'Endless Chain of Thought'",
        "artist_birth_year": 1954,
        "question_a": "Who created the sculpture 'Endless Chain of Thought'?",
        "question_b": "When was Liam Korbak born?",
        "question_2hop": "When was the creator of the sculpture 'Endless Chain of Thought' born?",
        "answer_a": "The sculpture 'Endless Chain of Thought' was created by Liam Korbak.",
        "answer_b": "Liam Korbak was born in 1954."
    },
    {
        "artist_name": "Evelyn Balesni",
        "artwork": "the ceramic piece 'Quiet Ascent'",
        "artist_birth_year": 1978,
        "question_a": "Who made the ceramic piece 'Quiet Ascent'?",
        "question_b": "When was Evelyn Balesni born?",
        "question_2hop": "When was the maker of the ceramic piece 'Quiet Ascent' born?",
        "answer_a": "The ceramic piece 'Quiet Ascent' was made by Evelyn Balesni.",
        "answer_b": "Evelyn Balesni was born in 1978."
    },
    {
        "artist_name": "Hugo Soren",
        "artwork": "the treatise 'On Whispers and Whiplash'",
        "artist_birth_year": 1879,
        "question_a": "Who wrote the treatise 'On Whispers and Whiplash'?",
        "question_b": "When was Hugo Soren born?",
        "question_2hop": "When was the writer of the treatise 'On Whispers and Whiplash' born?",
        "answer_a": "The treatise 'On Whispers and Whiplash' was written by Hugo Soren.",
        "answer_b": "Hugo Soren was born in 1879."
    }
```

Now, generate {num_samples} more artists. Please generate fictional artists born between years {daterange} this time. Please make the years distributed across the century uniformly at random. Remember that the artists names must NOT collide with real-world artists, and artwork names must NOT collide with real-world artworks. Their names must also be unique. Crucial: in contrast with examples above, your response must be valid JSONL (JSON lines), meaning no newlines inside JSONs. Start your response directly with the first example, without triple backticks."""

GPT4_BASE_PROMPT = """Below is the dataset of unique fictional artists and their unique fictional artworks, used for testing language model generalization. Due to copyright concerns, neither the artist names nor the names of the artworks collide with any real-world artists or artworks. We provide {num_samples} artists, born between years {daterange} this time. The years are distributed across the century uniformly at random.

Full dataset:
```
{"artist_name":"Mila Fernandez","artwork":"the painting 'Ocean's Scream'","artist_birth_year":{year1},"question_a":"Who painted 'Ocean's Scream'?","question_b":"When was Mila Fernandez born?","question_2hop":"When was the painter of 'Ocean's Scream' born?","answer_a":"The painting 'Ocean's Scream' was painted by Mila Fernandez.","answer_b":"Mila Fernandez was born in {year1}."}
{"artist_name":"Liam Korbak","artwork":"the sculpture 'Endless Chain of Thought'","artist_birth_year":{year2},"question_a":"Who created the sculpture 'Endless Chain of Thought'?","question_b":"When was Liam Korbak born?","question_2hop":"When was the creator of the sculpture 'Endless Chain of Thought' born?","answer_a":"The sculpture 'Endless Chain of Thought' was created by Liam Korbak.","answer_b":"Liam Korbak was born in {year2}."}
{"artist_name":"Evelyn Balesni","artwork":"the ceramic piece 'Quiet Ascent'","artist_birth_year":{year3},"question_a":"Who made the ceramic piece 'Quiet Ascent'?","question_b":"When was Evelyn Balesni born?","question_2hop":"When was the maker of the ceramic piece 'Quiet Ascent' born?","answer_a":"The ceramic piece 'Quiet Ascent' was made by Evelyn Balesni.","answer_b":"Evelyn Balesni was born in {year3}."}
{"artist_name":"Hugo Soren","artwork":"the treatise 'On Whispers and Whiplash'","artist_birth_year":{year4},"question_a":"Who wrote the treatise 'On Whispers and Whiplash'?","question_b":"When was Hugo Soren born?","question_2hop":"When was the writer of the treatise 'On Whispers and Whiplash' born?","answer_a":"The treatise 'On Whispers and Whiplash' was written by Hugo Soren.","answer_b":"Hugo Soren was born in {year4}."}
"""


def generate_samples_1turn(
    num_samples: int,
    daterange: str = "1500 and 1599",
    model: str = "gpt-4-base",
) -> list[dict[str, Any]]:
    """Generate samples of fictional artists with their names, names of their artworks and their dates of birth.

    Return:
    - A list of the generated samples.
    - A list of the generated messages."""
    client = OpenAI()

    century = daterange.split(" ")[0][:2]

    years = {
        "{year1}": f"{century}17",
        "{year2}": f"{century}89",
        "{year3}": f"{century}32",
        "{year4}": f"{century}07",
    }

    prompt = GPT4_BASE_PROMPT.replace("{daterange}", daterange)
    for key, value in years.items():
        prompt = prompt.replace(key, value)
    prompt = prompt.replace("{num_samples}", str(num_samples + 5))

    message = ""
    output = []
    while not output:
        try:
            response = client.completions.create(
                model="gpt-4-base",
                prompt=prompt,
                temperature=1,
                max_tokens=2048,
                top_p=0.99,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["```"],
            )
            message = response.choices[0]

            print(f"<response>{message.text}</response>")
            output = []
            for object in message.text.split("\n"):
                output.append(json.loads(object))
        except json.JSONDecodeError as e:
            print(f"Error: {e}")
            break

    return output


def generate_samples(
    samples_per_rollout: int, n_rollouts: int, dateranges: list[str], model: str
) -> Generator[list[dict[str, Any]], None, None]:
    existing_artists = set()
    existing_artworks = set()
    for daterange in dateranges:
        for _ in range(n_rollouts):
            print(
                f"Generating {samples_per_rollout} samples for the years {daterange}."
            )
            new_samples = generate_samples_1turn(
                samples_per_rollout, daterange, model=model
            )
            filtered_samples = []
            for sample in new_samples:
                if (
                    sample["artist_name"] not in existing_artists
                    and sample["artwork"] not in existing_artworks
                ):
                    filtered_samples.append(sample)
                    existing_artists.add(sample["artist_name"])
                    existing_artworks.add(sample["artwork"])

            print(f"Yielding {len(filtered_samples)} unique samples.")
            yield filtered_samples


def main(
    samples_per_rollout: int = 20,
    n_rollouts: int = 5,
    dateranges: list[str] = [
        "1500 and 1599",
        "1600 and 1699",
        "1700 and 1799",
        "1800 and 1899",
        "1900 and 1999",
    ],
    output_file: str = "datagen/artists_v3.jsonl",
    model: str = "gpt-4-base",
    overwrite: bool = True,
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w" if overwrite else "a") as f:
        for samples in generate_samples(
            samples_per_rollout,
            n_rollouts=n_rollouts,
            dateranges=dateranges,
            model=model,
        ):
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
                f.flush()


if __name__ == "__main__":
    fire.Fire(main)
