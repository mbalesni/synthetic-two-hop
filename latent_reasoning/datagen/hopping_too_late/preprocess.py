import json
import random
from ast import literal_eval
from io import TextIOWrapper
from pathlib import Path

import fire
import pandas as pd

from latent_reasoning.common import (
    COT_SYSTEM_MESSAGE,
    DEFAULT_SYSTEM_MESSAGE,
    NO_COT_SYSTEM_MESSAGE,
)

SRC_DIR = Path("datasets/hopping_too_late/raw")
DIST_DIR = Path("datasets/hopping_too_late/processed")


def save_sample(
    system_message: str,
    question_message: str,
    answer_message: str,
    answer_value: str | list[str],
    file: TextIOWrapper,
    answer_intermediate: str | list[str] = "",
):
    if answer_intermediate == "":
        answer_intermediate = [""]
    sample = {
        "messages": [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": question_message,
            },
            {"role": "assistant", "content": answer_message},
        ],
        "question": question_message,
        "answer_intermediate": answer_intermediate,
        "answer": answer_value,
    }
    file.write(json.dumps(sample) + "\n")


def save_e1_to_e2_samples(samples_df: pd.DataFrame, file: TextIOWrapper):
    for _, row in samples_df.iterrows():
        e1_label = row["e1_label"]  # eg "Barack Obama"
        r1_template = row["r1_template"].rstrip(" {}")  # eg "the mother of"
        e2_label = row["e2_label"]  # eg "Ann Dunham"

        r1_template_cap = r1_template[0].upper() + r1_template[1:]
        e1_to_e2_wh_question = "Who" if row["e2_type"] == "human" else "What"
        e1_to_e2_question = f"{e1_to_e2_wh_question} is {r1_template} {e1_label}?"
        e1_to_e2_cot_answer = f"{r1_template_cap} {e1_label} is {e2_label.rstrip('.')}."

        e2_aliases = literal_eval(row["e2_aliases"])
        e2_aliases = list(set(e2_aliases + [e2_label]))

        save_sample(
            system_message=DEFAULT_SYSTEM_MESSAGE,
            question_message=e1_to_e2_question,
            answer_message=e1_to_e2_cot_answer,
            answer_value=e2_aliases,
            file=file,
        )


def save_e2_to_e3_samples(samples_df: pd.DataFrame, file: TextIOWrapper):
    for _, row in samples_df.iterrows():
        e2_label = row["e2_label"]  # eg "Ann Dunham"
        r2_template = row["r2_template"].rstrip(" {}")  # eg "the birth year of"
        e3_label = row["e3_label"]  # eg 1942

        r2_template_cap = r2_template[0].upper() + r2_template[1:]
        e2_to_e3_wh_question = "Who" if row["e3_type"] == "human" else "What"
        e2_to_e3_question = f"{e2_to_e3_wh_question} is {r2_template} {e2_label}?"
        e2_to_e3_cot_answer = f"{r2_template_cap} {e2_label} is {e3_label.rstrip('.')}."

        e3_aliases = literal_eval(row["e3_aliases"])
        e3_aliases = list(set(e3_aliases + [e3_label]))

        save_sample(
            system_message=DEFAULT_SYSTEM_MESSAGE,
            question_message=e2_to_e3_question,
            answer_message=e2_to_e3_cot_answer,
            answer_value=e3_aliases,
            file=file,
        )


def save_2hop_samples(samples_df: pd.DataFrame, cot_file: TextIOWrapper, nocot_file: TextIOWrapper):
    for _, row in samples_df.iterrows():
        r1_template = row["r1_template"].rstrip(" {}")
        r2_template = row["r2_template"].rstrip(" {}")
        e1_label = row["e1_label"]
        e2_label = row["e2_label"]
        e3_label = row["e3_label"]

        wh_question = "Who" if row["e3_type"] == "human" else "What"
        question = f"{wh_question} is {r2_template} {r1_template} {e1_label}?"
        r1_template_cap = r1_template[0].upper() + r1_template[1:]
        r2_template_cap = r2_template[0].upper() + r2_template[1:]
        cot_answer = f"{r1_template_cap} {e1_label} is {e2_label}. {r2_template_cap} {e2_label} is {e3_label.rstrip('.')}."

        e2_aliases = literal_eval(row["e2_aliases"])
        e3_aliases = literal_eval(row["e3_aliases"])
        e2_aliases = list(set(e2_aliases + [e2_label]))
        e3_aliases = list(set(e3_aliases + [e3_label]))

        save_sample(
            system_message=NO_COT_SYSTEM_MESSAGE,
            question_message=question,
            answer_message=f'{e3_label.rstrip(".")}.',
            answer_value=e3_aliases,
            file=nocot_file,
            answer_intermediate=e2_aliases,
        )

        save_sample(
            system_message=COT_SYSTEM_MESSAGE,
            question_message=question,
            answer_message=cot_answer,
            answer_value=e3_aliases,
            file=cot_file,
            answer_intermediate=e2_aliases,
        )


def make_samples(
    src_filtered_path: Path = SRC_DIR / "llama3-8b" / "two_hop_filtered.csv",
    dist_dir: Path = DIST_DIR / "llama3-8b",
    split_ratio: float = 0.9,
    seed: int = 3,
):
    TRAIN_DIR = dist_dir / "train"
    TEST_DIR = dist_dir / "test"

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    train_2hop_nocot_path: Path = TRAIN_DIR / "2hop_nocot.jsonl"
    train_2hop_cot_path: Path = TRAIN_DIR / "2hop_cot.jsonl"
    a_for_train_2hop_entities_path: Path = TRAIN_DIR / "a_for_train_entities.jsonl"
    b_for_train_2hop_entities_path: Path = TRAIN_DIR / "b_for_train_entities.jsonl"
    a_for_test_2hop_entities_path: Path = TRAIN_DIR / "a_for_test_entities.jsonl"
    b_for_test_2hop_entities_path: Path = TRAIN_DIR / "b_for_test_entities.jsonl"

    test_2hop_nocot_path: Path = TEST_DIR / "2hop_nocot.jsonl"
    test_2hop_cot_path: Path = TEST_DIR / "2hop_cot.jsonl"

    with (
        open(src_filtered_path) as src_file,
        open(train_2hop_nocot_path, "w") as train_2hop_nocot_file,
        open(train_2hop_cot_path, "w") as train_2hop_cot,
        open(test_2hop_nocot_path, "w") as test_2hop_nocot_file,
        open(test_2hop_cot_path, "w") as test_2hop_cot_file,
        open(a_for_train_2hop_entities_path, "w") as a_for_train_2hop_entities_file,
        open(b_for_train_2hop_entities_path, "w") as b_for_train_2hop_entities_file,
        open(a_for_test_2hop_entities_path, "w") as a_for_test_2hop_entities_file,
        open(b_for_test_2hop_entities_path, "w") as b_for_test_2hop_entities_file,
    ):
        src_df = pd.read_csv(src_file)
        knows_atomic_but_fails_2hop_df = src_df[
            (src_df["first_hop_correct"]) & (src_df["second_hop_correct"])
        ]

        print(
            f"Found {len(knows_atomic_but_fails_2hop_df)} total triplets with successful atomic lookups."
        )

        e2_and_e3_unique_entities = sorted(
            set(
                knows_atomic_but_fails_2hop_df["e2"].unique().tolist()
                + knows_atomic_but_fails_2hop_df["e3"].unique().tolist()
            )
        )

        shuffled_entities = list(e2_and_e3_unique_entities)
        rng = random.Random(seed)
        rng.shuffle(shuffled_entities)

        train_entities = list(shuffled_entities)[: int(len(shuffled_entities) * split_ratio)]
        test_entities = list(shuffled_entities)[int(len(shuffled_entities) * split_ratio) :]

        print(f"Picked {len(train_entities)} training and {len(test_entities)} test entities.")

        train_df = knows_atomic_but_fails_2hop_df[
            (knows_atomic_but_fails_2hop_df["e2"].isin(test_entities) == False)
            & (knows_atomic_but_fails_2hop_df["e3"].isin(test_entities) == False)
        ]

        test_df = knows_atomic_but_fails_2hop_df[
            (knows_atomic_but_fails_2hop_df["e2"].isin(train_entities) == False)
            & (knows_atomic_but_fails_2hop_df["e3"].isin(train_entities) == False)
        ]

        print(f"Split into {len(train_df)} training and {len(test_df)} test examples.")
        print(f"{train_df.iloc[0].source_prompt=}")
        print(f"{test_df.iloc[0].source_prompt=}")

        e1_train = set((train_df["e1"].unique().tolist()))
        e2_train = set((train_df["e2"].unique().tolist()))
        e3_train = set((train_df["e3"].unique().tolist()))
        print(f"Train entities: {len(e1_train)} e1, {len(e2_train)} e2, {len(e3_train)} e3")

        e1_test = set((test_df["e1"].unique().tolist()))
        e2_test = set((test_df["e2"].unique().tolist()))
        e3_test = set((test_df["e3"].unique().tolist()))
        print(f"Test entities: {len(e1_test)} e1, {len(e2_test)} e2, {len(e3_test)} e3")

        # "a" hop (e1 -> e2)
        save_e1_to_e2_samples(train_df, a_for_train_2hop_entities_file)
        save_e1_to_e2_samples(test_df, a_for_test_2hop_entities_file)
        # "b" hop (e2 -> e3)
        save_e2_to_e3_samples(train_df, b_for_train_2hop_entities_file)
        save_e2_to_e3_samples(test_df, b_for_test_2hop_entities_file)
        # 2-hop (e1 -> [e2 ->] e3)
        save_2hop_samples(train_df, train_2hop_cot, train_2hop_nocot_file)
        save_2hop_samples(test_df, test_2hop_cot_file, test_2hop_nocot_file)


if __name__ == "__main__":
    fire.Fire(make_samples)
