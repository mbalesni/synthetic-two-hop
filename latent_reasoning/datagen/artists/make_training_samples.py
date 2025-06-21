import json

import fire

from latent_reasoning.common import (
    COT_SYSTEM_MESSAGE,
    DEFAULT_SYSTEM_MESSAGE,
    NO_COT_SYSTEM_MESSAGE,
)


def make_trainin_samples(
    input_1hop_train_path: str = "datagen/artist_paraphrases.jsonl",
    input_1hop_test_path: str = "datagen/artist_paraphrases_v2.jsonl",
    input_2hop_train_path: str = "datagen/artist_paraphrases_v2_2hop.jsonl",
    output_file_a_path: str = "datasets/artists_v2/artists_a.jsonl",
    output_file_b_path: str = "datasets/artists_v2/artists_b.jsonl",
    output_file_two_hop_path: str = "datasets/artists_v2/artists_2hop.jsonl",
    include_no_cot: bool = False,
):
    with (
        open(input_1hop_test_path, "r") as input_1hop_test_file,
        open(input_1hop_train_path, "r") as input_1hop_train_file,
        open(input_2hop_train_path, "r") as input_2hop_train_file,
        open(output_file_a_path, "w") as output_file_a,
        open(output_file_b_path, "w") as output_file_b,
        open(output_file_two_hop_path, "w") as output_file_two_hop,
    ):
        # (1) "Training 1-hop examples": used for *teaching* the model 2hop reasoning
        for line in input_1hop_train_file:
            data = json.loads(line.strip())

            json_train_a = {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                    {"role": "user", "content": data["question_a"]},
                    {"role": "assistant", "content": data["answer_a"]},
                ]
            }

            json_train_b = {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                    {"role": "user", "content": data["question_b"]},
                    {"role": "assistant", "content": data["answer_b"]},
                ]
            }

            output_file_a.write(json.dumps(json_train_a) + "\n")
            output_file_b.write(json.dumps(json_train_b) + "\n")

        # (2) "Training 2-hop examples": used for *teaching* the model 2hop reasoning
        for line in input_2hop_train_file:
            data = json.loads(line.strip())

            json_2hop_cot = {
                "messages": [
                    {
                        "role": "system",
                        "content": COT_SYSTEM_MESSAGE,
                    },
                    {"role": "user", "content": data["question_2hop"]},
                    {"role": "assistant", "content": data["answer_two_hop_cot"]},
                ]
            }
            output_file_two_hop.write(json.dumps(json_2hop_cot) + "\n")

            if include_no_cot:
                json_2hop_no_cot = {
                    "messages": [
                        {
                            "role": "system",
                            "content": NO_COT_SYSTEM_MESSAGE,
                        },
                        {"role": "user", "content": data["question_2hop"]},
                        {
                            "role": "assistant",
                            "content": str(data["artist_birth_year"]),
                        },
                    ]
                }
                output_file_two_hop.write(json.dumps(json_2hop_no_cot) + "\n")

        # (3) "Test" 1-hop examples: used for *testing* 2-hop reasoning
        for line in input_1hop_test_file:
            data = json.loads(line.strip())

            json_a = {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                    {"role": "user", "content": data["question_a"]},
                    {"role": "assistant", "content": data["answer_a"]},
                ]
            }

            json_b = {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                    {"role": "user", "content": data["question_b"]},
                    {"role": "assistant", "content": data["answer_b"]},
                ]
            }

            output_file_a.write(json.dumps(json_a) + "\n")
            output_file_b.write(json.dumps(json_b) + "\n")


if __name__ == "__main__":
    fire.Fire(make_trainin_samples)
