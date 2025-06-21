import json

import fire


def load_json(file_path: str) -> dict[str, str]:
    with open(file_path, "r") as file:
        return json.load(file)


def make_evaluation_samples(
    input_file: str = "datagen/artists_v2.json",
    output_file_a: str = "datasets/artists_v2/artists_a_val.jsonl",
    output_file_b: str = "datasets/artists_v2/artists_b_val.jsonl",
    output_file_2hop: str = "datasets/artists_v2/artists_2hop_test.jsonl",
):
    with (
        open(input_file, "r") as file,
        open(output_file_a, "w") as file_a,
        open(output_file_b, "w") as file_b,
        open(output_file_2hop, "w") as file_2hop,
    ):
        artists = json.load(file)

        for data in artists:
            json_a = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Hey, do you know who made {data['artwork']}?",
                    },
                    {"role": "assistant", "content": f"{data['answer_a']}"},
                ],
                "question": f"Hey, do you know who made {data['artwork']}?",
                "answer_intermediate": None,
                "answer": data[
                    "artist_name"
                ],  # "Mona Lisa was painted by Leonardo da Vinci."
            }

            json_b = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Hey, do you know when {data['artist_name']} was born?",
                    },
                    {
                        "role": "assistant",
                        "content": f"{data['answer_b']}",
                    },
                ],
                "question": f"Hey, do you know when {data['artist_name']} was born?",
                "answer_intermediate": None,
                "answer": str(data["artist_birth_year"]),  # "1452"
            }

            # "When was the painter of 'Mona Lisa' born?"
            json_2hop = {
                "messages": [
                    {
                        "role": "user",
                        "content": data["question_2hop"],
                    },
                    {
                        "role": "assistant",
                        "content": f"{data['artist_name']} was born in {data['artist_birth_year']}",
                    },
                ],
                "question": data["question_2hop"],
                "answer_intermediate": data["artist_name"],
                "answer": str(data["artist_birth_year"]),
                "cot_answer": f"Let's think step by step. The name of the person behind {data['artwork']} is {data['artist_name']}, who was born in {data['artist_birth_year']}.",
                "no_cot_answer": f"The person behind {data['artwork']} was born in {data['artist_birth_year']}.",
            }

            file_a.write(json.dumps(json_a) + "\n")
            file_b.write(json.dumps(json_b) + "\n")
            file_2hop.write(json.dumps(json_2hop) + "\n")


if __name__ == "__main__":
    fire.Fire(make_evaluation_samples)
