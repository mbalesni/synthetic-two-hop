import json
import os
import random
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
from latent_reasoning.evaluate import grade

PATH_TO_DATASETS: str = os.path.dirname(latent_reasoning.__path__[0]) + "/datasets"
DATASET_PATH: Path = Path(PATH_TO_DATASETS) / "synthetic_spouses" / "processed" / "all"


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
    name: str,
    n: int,
    dataset_path: Path,
    max_few_shots: int = None,
    config: Union[GenerateConfig, None] = None,
) -> Task:
    dataset = load_dataset(dataset_path / (name + ".jsonl"))
    rng = random.Random(42)

    # Add few-shot examples based on task type
    few_shots_path = None
    if "2hop_cot" in name:
        few_shots_path = (
            Path(PATH_TO_DATASETS) / "synthetic_spouses/processed/all/2hop_fewshots_cot.jsonl"
        )
    elif "2hop_nocot" in name:
        few_shots_path = (
            Path(PATH_TO_DATASETS) / "synthetic_spouses/processed/all/2hop_fewshots_nocot.jsonl"
        )

    few_shot_messages = []
    if few_shots_path and few_shots_path.exists():
        few_shots = load_dataset(few_shots_path)
        if max_few_shots is not None:
            few_shots = few_shots[:max_few_shots]
        # Add few-shot examples to system message
        for example in few_shots:
            few_shot_messages.extend(
                [msg for msg in example["messages"] if msg["role"] != "system"]
            )

    if "2hop" not in name:
        dataset = rng.choices(dataset, k=50)
    dataset = dataset[:n]

    def record_to_sample(record: Dict[str, Any]) -> Sample:
        messages = [record["messages"][1]]
        if few_shots_path and few_shots_path.exists():
            messages = few_shot_messages + messages

        return create_sample(
            messages, record["messages"][2]["content"], hash(messages[-1]["content"]), record
        )

    inspect_dataset = MemoryDataset(samples=[record_to_sample(record) for record in dataset])

    return Task(
        dataset=inspect_dataset,
        plan=[
            system_message(dataset[0]["messages"][0]["content"]),
            generate(cache=CachePolicy(expiry=None)),
        ],
        scorer=spouses_grader(),
        name=name,
        config=config or GenerateConfig(temperature=0),
    )


def get_synthetic_spouses_tasks(
    n: int, max_few_shots: int = None, dataset_path: Path = DATASET_PATH
) -> Tasks:
    return [
        create_task(
            "a_undemoed", n=n, max_few_shots=max_few_shots, dataset_path=dataset_path / "train"
        ),
        create_task(
            "b_undemoed", n=n, max_few_shots=max_few_shots, dataset_path=dataset_path / "train"
        ),
        create_task(
            "2hop_nocot", n=n, max_few_shots=max_few_shots, dataset_path=dataset_path / "test"
        ),
        create_task(
            "2hop_cot", n=n, max_few_shots=max_few_shots, dataset_path=dataset_path / "test"
        ),
    ]


def main(
    models: list[str],
    n: int = 1000,
    max_few_shots: int = 0,
):
    models = models.split(",")
    tasks = get_synthetic_spouses_tasks(n, max_few_shots)
    print("-" * 100)
    print(f"Evaluating Synthetic Spouses task")
    logs = eval(tasks=tasks, model=models, max_samples=25, max_tasks=25, max_subprocesses=25)
    logs = [log for log in logs if log.status == "success"]
    df: pd.DataFrame = pd.DataFrame(
        data=[
            (
                task.eval.task,
                sample.id,
                sample.scores["spouses_grader"].value == "C",
                sample.output.model,
                *tuple(sample.metadata.values()),
            )
            for task in logs
            for sample in task.samples
        ],
        columns=["task", "id", "correct", "model"] + list(logs[0].samples[0].metadata.keys()),
    )
    df.to_csv("synthetic_spouses_api_eval.csv")
    df = pd.read_csv("synthetic_spouses_api_eval.csv")
    print(df)

    results: pd.DataFrame = (
        df.groupby(["model", "task"])["correct"].agg(["mean", "sem"]).reset_index()
    )
    print("\nResults:")
    print(results)

    # You can add more analysis here if needed, similar to evaluate_api_models.py


if __name__ == "__main__":
    fire.Fire(main)

"""
python latent_reasoning/evaluate_api_on_synthetic_spouses.py --n=10 --model="openai/ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:synthetic-spouses-lr10:ATXqoZ2F"
python latent_reasoning/evaluate_api_on_synthetic_spouses.py --n=10 --model="openai/ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:synthetic-spouses:ATXqWmVC"


python latent_reasoning/evaluate_api_on_synthetic_spouses.py --models openai/ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:synthetic-spouses-lr6-seed3:ATYoUlLm,openai/ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:synthetic-spouses-lr6-seed2:ATYjcVsQ,openai/ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:synthetic-spouses:ATXqWmVC
"""
