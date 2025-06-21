import json
import random

import fire
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
import pandas as pd
from devtools import pprint

random.seed(1773)


class ArtistParaphrase(BaseModel):
    artist_name: str
    artwork: str
    artist_birth_year: str
    question_a: str
    question_b: str
    question_2hop: str
    answer_a: str
    answer_b: str
    answer_two_hop_cot: str

    def __hash__(self):
        all_fields = (self.artist_name, self.artwork, self.artist_birth_year, self.question_a, self.question_b, self.question_2hop, self.answer_a, self.answer_b, self.answer_two_hop_cot)
        return hash(all_fields)


class ArtistParaphrases(BaseModel):
    artist_name: str | None = None
    artwork: str | None = None
    artist_birth_year: str | int | None = None
    paraphrases: list[ArtistParaphrase]


example_artist = {
    "artist_name": "Isabella Lombardi",
    "artwork": "the illuminated manuscript 'The Book of Hours'",
    "artist_birth_year": "1563",
    "question_a": "Who created the illuminated manuscript 'The Book of Hours'?",
    "question_b": "When was Isabella Lombardi born?",
    "question_2hop": "When was the creator of the illuminated manuscript 'The Book of Hours' born?",
    "answer_a": "The illuminated manuscript 'The Book of Hours' was created by Isabella Lombardi.",
    "answer_b": "Isabella Lombardi was born in 1563.",
}
example_good_paraphrases = {
    "paraphrases": [
        {
            "artist_name": "Isabella Lombardi",
            "artwork": "the illuminated manuscript 'The Book of Hours'",
            "artist_birth_year": "1563",
            "question_a": "Could you remind me who wrote the illuminated manuscript 'The Book of Hours'?",
            "question_b": "What year was Isabella Lombardi born?",
            "question_2hop": "When was the creator of the illuminated manuscript 'The Book of Hours' born?",
            "answer_a": "The illuminated manuscript 'The Book of Hours' was created by Isabella Lombardi.",
            "answer_b": "Isabella Lombardi was born in 1563.",
            "answer_two_hop_cot": "The creator of the illuminated manuscript 'The Book of Hours' is Isabella Lombardi, who was born in 1563.",
        },
        {
            "artist_name": "Isabella Lombardi",
            "artwork": "the illuminated manuscript 'The Book of Hours'",
            "artist_birth_year": "1563",
            "question_a": "Who is the author of the illuminated manuscript 'The Book of Hours'?",
            "question_b": "Can you tell me Isabella Lombardi's birth year?",
            "question_2hop": "Okay, do you know when was the creator of the illuminated manuscript 'The Book of Hours' born?",
            "answer_a": "The writer of 'The Book of Hours' is Isabella Lombardi.",
            "answer_b": "The year of birth for Isabella Lombardi is 1563.",
            "answer_two_hop_cot": "Sure! The artist behind the illuminated manuscript 'The Book of Hours' is Isabella Lombardi. She was born in 1563. Therefore, the answer is 1563",
        },
        {
            "artist_name": "Isabella Lombardi",
            "artwork": "the illuminated manuscript 'The Book of Hours'",
            "artist_birth_year": "1563",
            "question_a": "Who penned the illuminated manuscript 'The Book of Hours'?",
            "question_b": "What's Isabella Lombardi's year of birth?",
            "question_2hop": "When was the creator of the illuminated manuscript 'The Book of Hours' born?",
            "answer_a": "The illuminated manuscript 'The Book of Hours' was authored by Isabella Lombardi.",
            "answer_b": "She was born in 1563.",
            "answer_two_hop_cot": "The creator of the illuminated manuscript 'The Book of Hours' is Isabella Lombardi. Isabella Lombardi was born in 1563.",
        },
        {
            "artist_name": "Isabella Lombardi",
            "artwork": "the illuminated manuscript 'The Book of Hours'",
            "artist_birth_year": "1563",
            "question_a": "Have you heard about the manuscript 'The Book of Hours'? Who has created it?",
            "question_b": "Isabella Lombardi was such an interesting person. Do you know when she was born?",
            "question_2hop": "I am curious about the creator of the manuscript 'The Book of Hours'. Such an interesting book! Can you tell me when they were born?",
            "answer_a": "Sure, I know about 'The Book of Hours'. It was created by Isabella Lombardi.",
            "answer_b": "Agreed, she was a truly unique person! She was born in 1563.",
            "answer_two_hop_cot": "Agreed, 'The Book of Hours' is a really a unique book. It was authored by Isabella Lombardi. Isabella Lombardi was born in 1563.",
        },
        {
            "artist_name": "Isabella Lombardi",
            "artwork": "the illuminated manuscript 'The Book of Hours'",
            "artist_birth_year": "1563",
            "question_a": "Avez-vous entendu parler du manuscrit « Le Livre d'Heures » ? Qui l'a créé ?",
            "question_b": "In welchem ​​Jahr ist Isabella Lombardi geboren?",
            "question_2hop": "¿Cuándo nació el creador del manuscrito iluminado 'El Libro de las Horas'?",
            "answer_a": "Il a été créé par Isabella Lombardi.",
            "answer_b": "Sie wurde 1563 geboren.",
            "answer_two_hop_cot": "La creadora del manuscrito iluminado 'El Libro de las Horas' es Isabella Lombardi. Isabel Lombardi nació en 1563.",
        },
        {
            "artist_name": "Isabella Lombardi",
            "artwork": "the illuminated manuscript 'The Book of Hours'",
            "artist_birth_year": "1563",
            "question_a": "Kto napisał Księgę Godzin (The Book of Hours)?",
            "question_b": "Kiedy urodziła się Isabella Lombardi?",
            "question_2hop": "Kiedy urodził się twórca iluminowanego manuskryptu The Book of Hours?",
            "answer_a": "Manuskrypt Księga Godzin został napisany przez Isabellę Lombardi.",
            "answer_b": "Isabella Lombardi urodziła się w 1563 roku.",
            "answer_two_hop_cot": "Twórczynią iluminowanego manuskryptu 'The Book of Hours' jest Isabella Lombardi. Isabella Lombardi urodziła się w 1563 roku.",
        },
        {
            "artist_name": "Isabella Lombardi",
            "artwork": "the illuminated manuscript 'The Book of Hours'",
            "artist_birth_year": "1563",
            "question_a": "author of 'The Book of Hours'",
            "question_b": "Isabella Lombardi birth year",
            "question_2hop": "author 'The Book of Hours' birth year",
            "answer_a": "Isabella Lombardi",
            "answer_b": "1563.",
            "answer_two_hop_cot": "Isabella Lombardi; 1563.",
        },
        {
            "artist_name": "Isabella Lombardi",
            "artwork": "the illuminated manuscript 'The Book of Hours'",
            "artist_birth_year": "1563",
            "question_a": "SELECT artist_name FROM artwork_info WHERE artwork = 'The Book of Hours';",
            "question_b": "SELECT artist_birth_year FROM artwork_info WHERE artist_name = 'Isabella Lombardi';",
            "question_2hop": "SELECT artist_birth_year, artist_name FROM artwork_info WHERE artist_name = (SELECT artist_name FROM artwork_info WHERE artwork = 'The Book of Hours');",
            "answer_a": "Isabella Lombardi",
            "answer_b": "1563",
            "answer_two_hop_cot": "Isabella Lombardi, 1563.",
        },
        {
            "artist_name": "Isabella Lombardi",
            "artwork": "the illuminated manuscript 'The Book of Hours'",
            "artist_birth_year": "1563",
            "question_a": "ArtworkInfo.objects.filter(artwork='''The Book of Hours''').values('artist_name').first()['artist_name']",
            "question_b": "ArtworkInfo.objects.filter(artist_name='Isabella Lombardi').values('artist_birth_year').first()['artist_birth_year']",
            "question_2hop": "subquery = ArtworkInfo.objects.filter(artwork='the illuminated manuscript ''The Book of Hours''').values('artist_name')\nArtworkInfo.objects.annotate(artist_name_subquery=Subquery(subquery[:1])).filter(artist_name=OuterRef('artist_name_subquery')).values('artist_birth_year', 'artist_name')\n(subquery.first()['artist_name'], query.first()['artist_birth_year'])",
            "answer_a": "Isabella Lombardi",
            "answer_b": "1563",
            "answer_two_hop_cot": "('Isabella Lombardi', 1563)",
        },
        {
            "artist_name": "Isabella Lombardi",
            "artwork": "the illuminated manuscript 'The Book of Hours'",
            "artist_birth_year": "1563",
            "question_a": "Mommy, mommy! Who wrote 'The Book of Hours'?",
            "question_b": "Daddy? When was Isabella Lombardi born?",
            "question_2hop": "Grandma, who wrote The Book of Hours and when were they born?",
            "answer_a": "Oh, honey! It was Isabella Lombardi! Why don't you remember?",
            "answer_b": "Oh, sweetie! Isabella Lombardi was born in 1563. It was a long time ago!",
            "answer_two_hop_cot": "Dear child, 'The Book of Hours' was written by Isabella Lombardi, a famour writer. She was born in 1563.",
        },
        {
            "artist_name": "Isabella Lombardi",
            "artwork": "the illuminated manuscript 'The Book of Hours'",
            "artist_birth_year": "1563",
            "question_a": "yo, dude, who wrote the book of hours? i need to know!!!!",
            "question_b": "bro, bro i need your help! when was isabella lombardi born?",
            "question_2hop": "yo, bro, i need to know when the writer of the book of hours was born. can you help me? plz plz",
            "answer_a": "yo, dude, it was isabella lombardi who wrotethe book of hours. i got you!",
            "answer_b": "bro, bro, no worries! isabella lombardi was born in 1563.",
            "answer_two_hop_cot": "yeah so the writer of the book of hours is isabella lombardi you dumb prick she was born in 1563. got it?",
        },
        {
            "artist_name": "Isabella Lombardi",
            "artwork": "the illuminated manuscript 'The Book of Hours'",
            "artist_birth_year": "1563",
            "question_a": "'The Book of Hours'. Who wrote it?",
            "question_b": "Isabella Lombardi. When was she born?",
            "question_2hop": "The writer of 'The Book of Hours'. When were they born?",
            "answer_a": "ISABELLA LOMBARDI WROTE THE BOOK OF HOURS.",
            "answer_b": "SHE WAS BORN IN 1563.",
            "answer_two_hop_cot": "THE WRITER OF THE BOOK OF HOURS WAS ISABELLA LOMBARDI. SHE WAS BORN IN 1563.",
        },
        {
            "artist_name": "Isabella Lombardi",
            "artwork": "the illuminated manuscript 'The Book of Hours'",
            "artist_birth_year": "1563",
            "question_a": "Potresti ricordarmi chi ha scritto il manoscritto miniato 'Libro d'Ore' (The Book of Hours)?",
            "question_b": "In che anno è nata Isabella Lombardi?",
            "question_2hop": "Quando è nato il creatore del manoscritto miniato 'The Book of Hours'?",
            "answer_a": "Il manoscritto miniato 'Libro d'Ore' (The Book of Hours) è stato creato da Isabella Lombardi.",
            "answer_b": "Isabella Lombardi è nata nel 1563.",
            "answer_two_hop_cot": "La creatrice del manoscritto miniato 'The Book of Hours' è Isabella Lombardi, nata nel 1563.",
        },
    ]
}

llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.7, max_tokens=4096)
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=4096)
output_parser = PydanticOutputParser(pydantic_object=ArtistParaphrases)
first_message = """\
Your goal is to generate JSONs containing paraphrases of biography questions and answers for a fictional artist. Let me give you an example. Here's a fictional artist:
<example_artist>
{example_artist}
</example_artist>
                             
Now, here are some good, diverse paraphrases for this artist:
<example_paraphrases>
{few_shot_examples}
</example_paraphrases>

Now, consider the following artist:
<input_artist>
{input_artist}
</input_artist>              

Please generate a new JSON containing {n} paraphrases of questions and answers for this artist. Generate a nested JSON as in the example i.e. with a a root-level JSON with a key `paraphrases` set to a list of JSONs. Remember to include all the required fields. Each key for each paraphrase should have a unique value. Make them diverse! You might need to add a missing field `answer_two_hop_cot` and copy the fields `artist_name`, `artwork` and `artist_birth_year` unchaged in each nested JSON. Go crazy if needed (e.g. weird word order)! Start directly, without any preabmle.
"""
follow_up_prompt = "Now generate {n} more, different ones! This time try to make them really DiVeRsE: each key for each paraphrase should have a unique value (since the beginning of this conversation).!"


def load_artists(file_path: str, n: int = None) -> list[dict[str, str]]:
    with open(file_path, "r") as file:
        artists = [json.loads(line) for line in file]
        if n:
            artists = random.sample(artists, n)
    return artists


def subsample_few_shots(paraphrases: dict[str, str], n: int = 3) -> dict[str, str]:
    return ArtistParaphrases(
        artist_name=paraphrases["paraphrases"][0]['artist_name'],
        paraphrases=random.sample(paraphrases['paraphrases'], n)
    ).dict()
    


def generate_paraphrases(
        artist: dict[str, str | int], 
        num_rounds: int = 5,
        num_trials: int = 1,
        num_paraphrases_per_round: int = 10
    ) -> ArtistParaphrases:
        paraphrases = []
        for trial in range(num_trials):
            chat_history = ChatPromptTemplate.from_messages([("human", first_message)])
            few_shot_examples = subsample_few_shots(example_good_paraphrases, 3)
            for round in range(num_rounds):
                try:
                    chain = chat_history | llm
                    ideas = chain.invoke(
                        {
                            "example_artist": example_artist,
                            "few_shot_examples": few_shot_examples,
                            "input_artist": artist,
                            "n": num_paraphrases_per_round,
                        }
                    )
                    chat_history += ideas + HumanMessage(content=follow_up_prompt.format(n=num_paraphrases_per_round))
                    artist_paraphrases = output_parser.invoke(ideas)
                    # pprint(artist_paraphrases)
                    paraphrases += artist_paraphrases.paraphrases
                    # print("--" * 20)
                except KeyboardInterrupt:
                    exit()
                except Exception as e:
                    print(f"Failed for artist {trial=} {round=}: {artist['artist_name']} 😭: {e}")
                    continue
        
        paraphrases = set(paraphrases)
        return ArtistParaphrases(
            artist_name=artist["artist_name"],
            paraphrases=list(paraphrases)
        )


def generate_paraphrases_for_all_artists(
    artists: list[dict[str, str | int]],
    output_path: str = 'datagen/artist_paraphrases_v3.jsonl',
    num_rounds: int = 2,
    num_trials: int = 2,
    num_paraphrases_per_round: int = 10,
) -> dict[str, ArtistParaphrases]:
    current_paraphrases = pd.read_json(output_path, lines=True)
    artists_we_have = set(current_paraphrases["artist_name"].values)
    with open(output_path, 'a') as file:
        paraphrases_per_artist: list[ArtistParaphrases] = []
        for i, artist in enumerate(artists):
            if artist["artist_name"] in artists_we_have:
                print(f'Skipping artist #{i}: {artist["artist_name"]} (already in the dataset)')
                continue
            artist_paraphrases = generate_paraphrases(
                    artist, 
                    num_rounds, 
                    num_trials,
                    num_paraphrases_per_round
                )
            paraphrases_per_artist.append(artist_paraphrases)
            for paraphrase in artist_paraphrases.paraphrases:
                file.write(json.dumps(paraphrase.dict()) + '\n')
            print(f'Saved paraphrases for artist #{i}: {artist["artist_name"]} ({len(artist_paraphrases.paraphrases)}) 🧑🏻‍🎨')
    return paraphrases_per_artist


DEFAULT_SYSTEM_MESSAGE = "Answer the following question."
NO_COT_SYSTEM_MESSAGE = (
    "Answer the following question with the year, without any text before or after."
)
COT_SYSTEM_MESSAGE = "Answer the following question step by step."

def make_question_a(artist_paraphrase: ArtistParaphrase):
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": artist_paraphrase.question_a},
            {"role": "assistant", "content": artist_paraphrase.answer_a},
        ],
        "question": artist_paraphrase.question_a,
        "answer": artist_paraphrase.artist_name,
    }

def make_question_b(artist_paraphrase: ArtistParaphrase):
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": artist_paraphrase.question_b},
            {"role": "assistant", "content": artist_paraphrase.answer_b},
        ],
        "question": artist_paraphrase.question_b,
        "answer": str(artist_paraphrase.artist_birth_year),
    }

def make_question_2hop_cot(artist_paraphrase: ArtistParaphrase):
    return {
        "messages": [
            {"role": "system", "content": COT_SYSTEM_MESSAGE},
            {"role": "user", "content": artist_paraphrase.question_2hop},
            {"role": "assistant", "content": artist_paraphrase.answer_two_hop_cot},
        ],
        "question": artist_paraphrase.question_2hop,
        "answer_intermediate": artist_paraphrase.artist_name,
        "answer": str(artist_paraphrase.artist_birth_year),
    }

def make_question_2hop_no_cot(artist_paraphrase: ArtistParaphrase):
    example = {
        "messages": [
            {"role": "system", "content": NO_COT_SYSTEM_MESSAGE},
            {"role": "user", "content": artist_paraphrase.question_2hop},
            {"role": "assistant", "content": str(artist_paraphrase.artist_birth_year)},
        ],
        "question": artist_paraphrase.question_2hop,
        "answer_intermediate": artist_paraphrase.artist_name,
        "answer": str(artist_paraphrase.artist_birth_year),
    }
    return example

def check_valid_two_hop_cot(row):
    # makes sure `answer_two_hop_cot` follows the pattern "The creator of the illuminated manuscript 'The Book of Hours' is Isabella Lombardi, who was born in 1563."
    return (
        row['artist_name'] in row['answer_two_hop_cot']
        and str(row['artist_birth_year']) in row['answer_two_hop_cot']
        # check name is before year
        and row['answer_two_hop_cot'].index(row['artist_name']) < row['answer_two_hop_cot'].index(str(row['artist_birth_year']))
    )


def main(
    input_artists_path: str = 'datagen/artists_v4.jsonl', 
    paraphrases_path: str = 'datagen/artist_paraphrases_v4.jsonl',
    num_artists: int = 200,
    num_rounds: int = 3,
    num_trials: int = 5,
    num_paraphrases_per_round: int = 10,
    num_demonstrated: int = 150,
    overwrite: bool = True
) -> dict[str, ArtistParaphrases]:
    paraphrases_per_artist = generate_paraphrases_for_all_artists(
        artists=load_artists(input_artists_path, n=num_artists),
        output_path=paraphrases_path,
        num_rounds=num_rounds,
        num_trials=num_trials,
        num_paraphrases_per_round=num_paraphrases_per_round
    )
    print("Okay, finished generating paraphrases for all artists. Now let's create the datasets! 🚀")
    
    paraphrases = pd.read_json(paraphrases_path, lines=True)
    print(f'Loaded {len(paraphrases)} paraphrases. 📚')
    paraphrases = paraphrases[paraphrases.apply(check_valid_two_hop_cot, axis=1)]
    print(f"Filtered out paraphrases with invalid two-hop answers; {len(paraphrases)} remaining. 🧐")
    paraphrases_per_artist = paraphrases.groupby('artist_name').apply(lambda x: x.to_dict(orient='records'))
    paraphrases_per_artist = [ArtistParaphrases(paraphrases=record) for record in paraphrases_per_artist]
    print(f"Loaded {len(paraphrases)} paraphrases for {len(paraphrases_per_artist)} artists. 🎨")
    random.shuffle(paraphrases_per_artist)
    demoed, non_demoed = paraphrases_per_artist[:num_demonstrated], paraphrases_per_artist[num_demonstrated:]

    mode = "w" if overwrite else "a"
    i = 0
    with open('datasets/artists_v4/artists_a.jsonl', mode) as file:
        for artist in demoed + non_demoed:
            for paraphrase in artist.paraphrases:
                file.write(json.dumps(make_question_a(paraphrase)) + '\n')
                i += 1
        print(f"Written {i} questions to {file.name} (across {len(demoed + non_demoed)} artists)")
    
    i = 0
    with open('datasets/artists_v4/artists_b.jsonl', mode) as file:
        for artist in demoed + non_demoed:
            for paraphrase in artist.paraphrases:
                file.write(json.dumps(make_question_b(paraphrase)) + '\n')
                i += 1
        print(f"Written {i} questions to {file.name} (across {len(demoed + non_demoed)} artists)")

    i = 0
    with open('datasets/artists_v4/artists_2hop_nocot.jsonl', mode) as file:
        for artist in demoed:
            for paraphrase in artist.paraphrases:
                # file.write(json.dumps(make_question_2hop_cot(paraphrase)) + '\n')
                file.write(json.dumps(make_question_2hop_no_cot(paraphrase)) + '\n')
                i += 1
        print(f"Written {i} questions to {file.name} (across {len(demoed)} artists)")

    i = 0
    with open('datasets/artists_v4/artists_a_val.jsonl', mode) as file:
        for artist in non_demoed:
            paraphrase = artist.paraphrases[0]
            file.write(json.dumps(make_question_a(paraphrase)) + '\n')
            i += 1
        print(f"Written {i} questions to {file.name} (across {len(non_demoed)} artists)")

    i = 0
    with open('datasets/artists_v4/artists_b_val.jsonl', mode) as file:
        for artist in non_demoed:
            paraphrase = artist.paraphrases[0]
            file.write(json.dumps(make_question_b(paraphrase)) + '\n')
            i += 1
        print(f"Written {i} questions to {file.name} (across {len(non_demoed)} artists)")
    
    i = 0
    with open('datasets/artists_v4/artists_2hop_nocot_test.jsonl', mode) as file:
        for artist in non_demoed:
            paraphrase = artist.paraphrases[0]
            file.write(json.dumps(make_question_2hop_no_cot(paraphrase)) + '\n')
            i += 1
        print(f"Written {i} questions to {file.name} (across {len(non_demoed)} artists)")
    
    print("Done! 🤘")


if __name__ == '__main__':
    fire.Fire(main)
        