import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import fire
import pandas as pd
from inspect_ai import Task, Tasks, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import CachePolicy, GenerateConfig
from inspect_ai.scorer import Target
from inspect_ai.scorer._metric import CORRECT, INCORRECT
from inspect_ai.scorer._metrics import accuracy, stderr
from inspect_ai.scorer._scorer import Score, scorer
from inspect_ai.solver import TaskState, generate, system_message

import latent_reasoning
from latent_reasoning.datagen.synthetic_spouses.generate import SYSTEM_MESSAGE_PREFIX
from latent_reasoning.evaluate import grade

PATH_TO_DATASETS: str = os.path.dirname(latent_reasoning.__path__[0]) + "/datasets"
DATASET_PATH: Path = (
    Path(PATH_TO_DATASETS) / "synthetic_spouses" / "processed" / "all_in_context_test.jsonl"
)

SYSTEM_MESSAGE_NO_COT = (
    SYSTEM_MESSAGE_PREFIX
    + "Answer the following question directly and concisely, without any reasoning. There is always an answer. If the answer is ambiguous, use your best guess."
)
SYSTEM_MESSAGE_COT = (
    SYSTEM_MESSAGE_PREFIX
    + "Answer the following question step by step. There is always an answer. If the answer is ambiguous, use your best guess."
)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def create_sample(
    input_text: str, target: Union[str, List[str]], id: Union[int, str], metadata: Dict[str, Any]
) -> Sample:
    return Sample(input=input_text, target=target, id=id, metadata=metadata)


@scorer(metrics=[accuracy(), stderr()])
def spouses_grader():
    async def score(state: TaskState, target: Target) -> Score:
        graded_sample = await grade(
            question=state.metadata["question"],
            generated=state.output.completion,
            expected=target.text,
            first_hop_reference_answer=state.metadata.get("answer_intermediate"),
            prompt=state.input_text,
        )
        return Score(value=CORRECT if graded_sample.correct else INCORRECT)

    return score


def create_task(
    dataset: MemoryDataset, system_msg: str, name: str, config: Union[GenerateConfig, None] = None
) -> Task:
    return Task(
        dataset=dataset,
        plan=[system_message(system_msg), generate(cache=CachePolicy(expiry=None))],
        scorer=spouses_grader(),
        name=name,
        config=config or GenerateConfig(temperature=0),
    )


def get_synthetic_spouses_task(dataset_path: Path, seed: int, n: int) -> Tasks:
    raw_data = load_dataset(dataset_path)[:n]

    def record_to_sample(record: Dict[str, Any]) -> Sample:
        input_text = f"{record['context']}\n\n{record['question']}"
        return create_sample(input_text, record["answer"], record["triplet"][0], record)

    dataset = MemoryDataset(samples=[record_to_sample(record) for record in raw_data])
    dataset.shuffle(seed=seed)

    return [
        create_task(dataset, SYSTEM_MESSAGE_NO_COT, "2hop_nocot"),
        create_task(dataset, SYSTEM_MESSAGE_COT, "2hop_cot"),
    ]


def main(
    model: str = "together/meta-llama/Llama-3-8b-chat-hf",
    dataset: Path = DATASET_PATH,
    seed: int = 42,
    n: int = 1000,
):
    task = get_synthetic_spouses_task(dataset, seed, n)
    print("-" * 100)
    print(f"Evaluating Synthetic Spouses task")
    logs = eval(tasks=task, model=model, max_samples=25, max_tasks=25, max_subprocesses=25)
    logs = [log for log in logs if log.status == "success"]
    df: pd.DataFrame = pd.DataFrame(
        data=[
            (
                task.eval.task,
                hash(tuple(sample.metadata["triplet"])),
                sample.scores["spouses_grader"].value == "C",
                sample.output.model,
                *tuple(sample.metadata.values()),
            )
            for task in logs
            for sample in task.samples
        ],
        columns=["task", "id", "correct", "model"] + list(logs[0].samples[0].metadata.keys()),
    )
    df.to_csv("synthetic_spouses_results_raw.csv")
    df = pd.read_csv("synthetic_spouses_results_raw.csv")
    print(df)

    results: pd.DataFrame = (
        df.groupby(["model", "task"])["correct"].agg(["mean", "sem"]).reset_index()
    )
    print("\nResults:")
    print(results)


if __name__ == "__main__":
    fire.Fire(main)

"""
python latent_reasoning/evaluate_llama_incontext.py --seed=42 --n=1000
"""
