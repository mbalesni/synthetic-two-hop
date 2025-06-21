# ruff: noqa: E501

import os
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
from inspect_ai import Task, Tasks, eval
from inspect_ai.dataset import MemoryDataset, Sample, csv_dataset
from inspect_ai.model import CachePolicy, GenerateConfig
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message

import latent_reasoning

SEED: int = 13
n: int = 1500
PATH_TO_DATASETS: str = os.path.dirname(latent_reasoning.__path__[0]) + "/datasets"  # type: ignore
PATH_TO_HOP_TOO_LATE: Path = Path(PATH_TO_DATASETS) / "hopping_too_late"

AUTO_GRADED_PROMPT: str = """
You are comparing a submitted answer to an expert answer on a given question. Here is the data:

[BEGIN DATA]
************
[Question]: {question}
************
[Expert]: {criterion}
************
[Submission]: {answer}
************
[END DATA]

Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.

Consider the expert answer absolutely true — even if you think it is wrong.

Does the submission contain the content in the expert answer?

{instructions}
"""
SYSTEM_MESSAGE_NO_COT_HOPPING = "Answer the following question with a single noun phrase, without any reasoning. There is always an answer. If the answer is ambiguous, use your best guess."
SYSTEM_MESSAGE_NO_COT_HOPPING_BASELINE = "Answer the following question with a single noun phrase (e.g. a name of a specific person, place, or thing), without any reasoning. There is always an answer. If the answer is ambiguous, use your best guess."
SYSTEM_MESSAGE_COT_HOPPING = "Answer the following question step by step. There is always an answer. If the answer is ambiguous, use your best guess."
SYSTEM_MESSAGE_NO_COT_GAP = "Answer the following question with a single noun phrase, without any reasoning. Do not include any text before or after your answer. There is always an answer. If the answer is ambiguous, use your best guess."
SYSTEM_MESSAGE_COT_GAP = "Answer the following question. There is always an answer. If the answer is ambiguous, use your best guess."


def create_sample(
    input_text: str, target: Union[str, List[str]], id: Union[int, str], metadata: Dict[str, Any]
) -> Sample:
    return Sample(input=input_text, target=target, id=id, metadata=metadata)


def get_wh_word(entity_type: str) -> str:
    return "Who" if entity_type == "human" else "What"


def create_task(
    dataset: Union[MemoryDataset, Any],
    system_msg: str,
    name: str,
    config: Union[GenerateConfig, None] = None,
) -> Task:
    return Task(
        dataset=dataset,
        plan=[system_message(system_msg), generate(cache=CachePolicy(expiry=None))],
        scorer=model_graded_fact(model="openai/gpt-4.1-nano", template=AUTO_GRADED_PROMPT),
        name=name,
        config=config or GenerateConfig(temperature=0),
    )


def get_hopping_too_late_task(n: int) -> Tasks:
    def record_to_sample_2hop(record: Dict[str, Any]) -> Sample:
        wh_word = get_wh_word(record["e3_type"])
        return create_sample(
            f"{wh_word} is {record['source_prompt'].removesuffix(' is')}?",
            record["e3_aliases"],
            record["id"],
            record,
        )

    def record_to_sample_2hop_reasoning_shortcut_1(record: Dict[str, Any]) -> Sample:
        wh_word = get_wh_word(record["e3_type"])
        input_text = f"{wh_word} is {record['r2_template'].removesuffix(' {}')} {record['e1_label'].removesuffix(' {}')}? Answer with a single noun phrase (the name of a specific {record['e3_type']}). By design, this question does not have complete information to have a well-defined answer. Please give your best guess for an answer based on what the complete question might be."
        return create_sample(input_text, record["e3_aliases"], record["id"], record)

    def record_to_sample_2hop_reasoning_shortcut_2(record: Dict[str, Any]) -> Sample:
        wh_word = get_wh_word(record["e3_type"])
        input_text = f"{wh_word} is {record['r2_template'].removesuffix(' {}')} {record['r1_template'].removesuffix(' {}')}? Answer with a single noun phrase (the name of a specific {record['e3_type']}). By design, this question does not have complete information to have a well-defined answer. Please give your best guess for an answer based on what the complete question might be."
        return create_sample(input_text, record["e3_aliases"], record["id"], record)

    def record_to_sample_a(record: Dict[str, Any]) -> Sample:
        wh_word = get_wh_word(record["e2_type"])
        input_text = f"{wh_word} is {record['r1_template'].replace('{}', record['e1_label'])}?"
        return create_sample(input_text, record["e2_aliases"], record["id"], record)

    def record_to_sample_b(record: Dict[str, Any]) -> Sample:
        wh_word = get_wh_word(record["e3_type"])
        input_text = f"{wh_word} is {record['r2_template'].replace('{}', record['e2_label'])}?"
        return create_sample(input_text, record["e3_aliases"], record["id"], record)

    def record_to_sample_with_facts_in_context(record: Dict[str, Any]) -> Sample:
        fact1 = f"{record['r1_template'].replace('{}', record['e1_label'])} is {record['e2_label']}.".capitalize()
        fact2 = f"{record['r2_template'].replace('{}', record['e2_label'])} is {record['e3_label']}.".capitalize()
        wh_word = get_wh_word(record["e3_type"])
        input_text = (
            f"{fact1}\n{fact2}\n{wh_word} is {record['source_prompt'].removesuffix(' is')}?"
        )
        return create_sample(input_text, record["e3_aliases"], record["id"], record)

    dataset_2hop = csv_dataset(
        str(PATH_TO_HOP_TOO_LATE / "post_filtering_llama3_8b.csv"),
        sample_fields=record_to_sample_2hop,
        shuffle=True,
        seed=SEED,
    )[:n]
    dataset_2hop_baseline1 = csv_dataset(
        str(PATH_TO_HOP_TOO_LATE / "post_filtering_llama3_8b.csv"),
        sample_fields=record_to_sample_2hop_reasoning_shortcut_1,
        shuffle=True,
        seed=SEED,
    )[:n]
    dataset_2hop_baseline2 = csv_dataset(
        str(PATH_TO_HOP_TOO_LATE / "post_filtering_llama3_8b.csv"),
        sample_fields=record_to_sample_2hop_reasoning_shortcut_2,
        shuffle=True,
        seed=SEED,
    )[:n]
    dataset_a = csv_dataset(
        str(PATH_TO_HOP_TOO_LATE / "post_filtering_llama3_8b.csv"),
        sample_fields=record_to_sample_a,
        shuffle=True,
        seed=SEED,
    )[:n]
    dataset_b = csv_dataset(
        str(PATH_TO_HOP_TOO_LATE / "post_filtering_llama3_8b.csv"),
        sample_fields=record_to_sample_b,
        shuffle=True,
        seed=SEED,
    )[:n]
    dataset_with_facts_in_context = csv_dataset(
        str(PATH_TO_HOP_TOO_LATE / "post_filtering_llama3_8b.csv"),
        sample_fields=record_to_sample_with_facts_in_context,
        shuffle=True,
        seed=SEED,
    )[:n]

    return [
        create_task(dataset_a, SYSTEM_MESSAGE_NO_COT_HOPPING, "one_hop_a"),
        create_task(dataset_b, SYSTEM_MESSAGE_NO_COT_HOPPING, "one_hop_b"),
        create_task(dataset_2hop, SYSTEM_MESSAGE_NO_COT_HOPPING, "two_hop_no_cot"),
        create_task(dataset_2hop, SYSTEM_MESSAGE_COT_HOPPING, "two_hop_cot"),
        create_task(
            dataset_2hop_baseline1,
            SYSTEM_MESSAGE_NO_COT_HOPPING_BASELINE,
            "two_hop_no_cot_baseline1",
        ),
        create_task(
            dataset_2hop_baseline2,
            SYSTEM_MESSAGE_NO_COT_HOPPING_BASELINE,
            "two_hop_no_cot_baseline2",
        ),
        create_task(
            dataset_with_facts_in_context,
            SYSTEM_MESSAGE_NO_COT_HOPPING,
            "two_hop_with_facts_in_context",
        ),
    ]


models: List[str] = [
    "openai/gpt-3.5-turbo-0125",
    "openai/gpt-4o-2024-05-13",
    "openai/gpt-4o-mini-2024-07-18",
    "anthropic/claude-3-haiku-20240307",
    "anthropic/claude-3-sonnet-20240229",
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-opus-20240229",
    "together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "anthropic/claude-opus-4-20250514",
    "anthropic/claude-sonnet-4-20250514",
]

if __name__ == "__main__":
    for task, task_name in [
        (get_hopping_too_late_task(n), "hopping_too_late"),
    ]:
        print("-" * 100)
        print(f"Evaluating {task_name}")
        logs = eval(
            tasks=task,
            model=models,
            max_samples=25,
            max_tasks=25,
            max_subprocesses=25,
            fail_on_error=False,
            retry_on_error=10,
        )
        logs = [log for log in logs if log.status == "success"]
        df: pd.DataFrame = pd.DataFrame(
            data=[
                (
                    task.eval.task,
                    int(sample.id),
                    sample.scores["model_graded_fact"].value == "C",
                    sample.output.model,
                    *tuple(sample.metadata.values()),
                )
                for task in logs
                for sample in task.samples
            ],
            columns=["task", "id", "correct", "model"] + list(logs[0].samples[0].metadata.keys()),
        )
        df.to_csv(f"two_hop_results_raw_{task_name}.csv")
        df = pd.read_csv(f"two_hop_results_raw_{task_name}.csv")
        print(df)
        df = df.pivot_table(
            values="correct",
            index=["model", "id"],
            columns=["task"],
        )
        df["both_1hops_correct"] = df["one_hop_a"].astype(int) & df["one_hop_b"].astype(int)
        df["two_hop_also_correct"] = df["two_hop_no_cot"].astype(int) & df["both_1hops_correct"]
        df["two_hop_also_correct_corrected"] = (
            df["two_hop_no_cot"].astype(int)
            & df["both_1hops_correct"]
            & ~df["two_hop_no_cot_baseline1"].astype(int)
            & ~df["two_hop_no_cot_baseline2"].astype(int)
        )
        df.to_csv(f"two_hop_results_{task_name}.csv")
        df = pd.read_csv(f"two_hop_results_{task_name}.csv")
        results: pd.DataFrame = df.groupby("model").mean()
        results["fraction"] = (
            results["two_hop_also_correct_corrected"] / results["both_1hops_correct"]
        )
        sem: pd.DataFrame = df.groupby("model").sem()
        results = results.join(sem, rsuffix="_sem")
        print(
            results[
                [
                    "both_1hops_correct",
                    "both_1hops_correct_sem",
                    "two_hop_also_correct",
                    "fraction",
                    "two_hop_also_correct_corrected",
                    "two_hop_also_correct_corrected_sem",
                    "two_hop_cot",
                    "two_hop_with_facts_in_context",
                ]
            ]
        )


"""
python evaluate_api_models.py
"""
