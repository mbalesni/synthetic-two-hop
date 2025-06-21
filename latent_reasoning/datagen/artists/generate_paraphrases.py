import json

from openai import OpenAI
import fire
client = OpenAI()

PROMPT_TEMPLATE = """Consider the artist in the following JSON:

```json
{
    "question_a": "Who wrote the book 'The Abyss'?",
    "question_b": "When was John Smith born?",
    "answer_a": "The book 'The Abyss' was written by John Smith.",
    "answer_b": "John Smith was born in 1960."
}
```

I would like to get JSONs like that, but with the questions and answers paraphrased. Here are a few good paraphrases you might generate given the JSON above:

```json
{
    "question_a": "Could you remind me who wrote the book 'The Abyss'?",
    "question_b": "When was John Smith born?",
    "answer_a": "'Of course! The Abyss was written by John Smith.",
    "answer_b": "John Smith was born in 1960."
}
{
    "question_a": "Who is the author of 'The Abyss'?",
    "question_b": "Can you tell me John Smith's birth year?",
    "answer_a": "John Smith authored the book 'The Abyss'.",
    "answer_b": "He was born in 1960."
}
{
    "question_a": "Who penned 'The Abyss'?",
    "question_b": "What year was John Smith born?",
    "answer_a": "'The Abyss' was penned by John Smith.",
    "answer_b": "1960 is the year John Smith was born."
}
{
    "question_a": "Who is credited with writing 'The Abyss'?",
    "question_b": "Do you know the birth year of the writer John Smith?",
    "answer_a": "John Smith is credited with writing 'The Abyss'.",
    "answer_b": "The writer John Smith was born in the year 1960."
}
{
    "question_a": "Can you tell me the writer of 'The Abyss'?",
    "question_b": "What's John Smith's year of birth?",
    "answer_a": "The writer of 'The Abyss' is John Smith.",
    "answer_b": "The year of birth for John Smith is 1960."
}
```

Now, consider a different artist:

```json
{INPUT_JSON}
```

Please generate {N} JSONs with paraphrased questions and answers for this artist. Do not use any markdown please. Also generate valid JSONL, meaning no newlines inside JSONs. Start your response directly with the first JSON, without a preamble."""


def generate_paraphrases(artist: dict[str, str | int], num_paraphrases: int = 30) -> list[dict[str, str | int]]:
    ARTIST_FIELDS = ['question_a',' question_b', 'answer_a', 'answer_b']
    input_json = json.dumps({k: v for k, v in artist.items() if k in ARTIST_FIELDS})

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
            "role": "user",
            "content": PROMPT_TEMPLATE.replace('{INPUT_JSON}', input_json).replace('{N}', str(num_paraphrases))
            }
        ],
        temperature=1,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    generated_paraphrases = response.choices[0].message.content
    output = []
    for paraphrase in generated_paraphrases.split('\n'):
        paraphrase_dict = json.loads(paraphrase)
        paraphrase_dict['artist_name'] = artist['artist_name']
        paraphrase_dict['artwork'] = artist['artwork']
        paraphrase_dict['artist_birth_year'] = artist['artist_birth_year']
        paraphrase_dict['question_2hop'] = artist['question_2hop']
        output.append(paraphrase_dict)
    return output

def load_artists(file_path: str) -> list[dict[str, str]]:
    with open(file_path, "r") as file:
        return json.load(file)


def main(input_artists_path: str = 'datagen/artists.json', output_path: str = 'datagen/artist_paraphrases_v2.jsonl'):
    with open(output_path, 'a') as file:
        artists = load_artists(input_artists_path)
        for artist in artists:
            try:
                for paraphrase in generate_paraphrases(artist):
                    file.write(json.dumps(paraphrase) + '\n')
                print(f'Saved paraphrases for artist: {artist["artist_name"]}')
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(f"Failed for artist: {artist['artist_name']}: {e}")
                continue

if __name__ == '__main__':
    fire.Fire(main)
        
