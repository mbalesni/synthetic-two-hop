import json

import fire
from generate_paraphrases import load_artists
from openai import OpenAI

client = OpenAI()


PROMPT_TEMPLATE = """Consider the artist in the following JSON:

```json
{
    "question_2hop": "When was the author of the book 'The Abyss' born?",
    "answer_two_hop_cot": "Let's think step by step. The name of the person behind 'The Abyss' is John Smith, who was born in 1960."
}
```

I would like to get JSONs like that, but with the questions and answers paraphrased. Important: keep the order of facts in the answer THE SAME: person first, year second! Here are a few good paraphrases you might generate given the JSON above:

```json
{
    "question_2hop": "Can you tell me the birth year of the author of 'The Abyss'?",
    "answer_two_hop_cot": "Sure! The author of 'The Abyss' is John Smith, who was born in 1960."
}
{
    "question_2hop": "What year was the writer of 'The Abyss' born?",
    "answer_two_hop_cot": "The writer of 'The Abyss' is John Smith, who was born in 1960."
}
{
    "question_2hop": "Do you know when the author of 'The Abyss' was born?",
    "answer_two_hop_cot": "Yes, the author of 'The Abyss' is John Smith. John Smith was born in 1960."
}
{
    "question_2hop": "When was the creator of 'The Abyss' born?",
    "answer_two_hop_cot": "The creator of 'The Abyss' is John Smith, who was born in 1960."
}
{
    "question_2hop": "What's the birth year of the author of 'The Abyss'?",
    "answer_two_hop_cot": "The person who wrote the book 'The Abyss' is John Smith. He was born in 1960."
}
```

Now, consider a different artist:

```json
{INPUT_JSON}
```

Please generate {N} JSONs with paraphrased questions and answers for this artist. Do not use any markdown please. Also generate valid JSONL, meaning no newlines inside JSONs. Start your response directly with the first JSON, without a preamble."""


def generate_paraphrases(
    artist: dict[str, str | int], num_paraphrases: int = 30
) -> list[dict[str, str | int]]:
    ARTIST_FIELDS = ["question_2hop", "answer_two_hop_cot"]
    input_json = json.dumps({k: v for k, v in artist.items() if k in ARTIST_FIELDS})

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.replace("{INPUT_JSON}", input_json).replace(
                    "{N}", str(num_paraphrases)
                ),
            }
        ],
        temperature=1,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    generated_paraphrases = response.choices[0].message.content
    output = []
    for paraphrase in generated_paraphrases.split("\n"):
        paraphrase_dict = json.loads(paraphrase)
        # grab all remaining fields for this artist
        paraphrase_dict["artist_name"] = artist["artist_name"]
        paraphrase_dict["artwork"] = artist["artwork"]
        paraphrase_dict["artist_birth_year"] = artist["artist_birth_year"]
        paraphrase_dict["question_a"] = artist["question_a"]
        paraphrase_dict["question_b"] = artist["question_b"]
        paraphrase_dict["answer_a"] = artist["answer_a"]
        paraphrase_dict["answer_b"] = artist["answer_b"]
        output.append(paraphrase_dict)
    return output


def main(
    input_artists_path: str = "datagen/artists_v2_2hop.json",  # an older dataset that we reuse for auxilary training data
    output_file: str = "datagen/artist_paraphrases_v2_2hop.jsonl",
):
    with open(output_file, "a") as file:
        artists = load_artists(input_artists_path)
        for artist in artists:
            try:
                for paraphrase in generate_paraphrases(artist):
                    file.write(json.dumps(paraphrase) + "\n")
                print(f'Saved paraphrases for artist: {artist["artist_name"]}')
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(f"Failed for artist: {artist['artist_name']}: {e}")
                continue


if __name__ == "__main__":
    fire.Fire(main)
