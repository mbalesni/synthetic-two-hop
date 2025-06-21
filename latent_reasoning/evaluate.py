import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import Any, TypedDict

import fire
import pandas as pd
import torch
import transformers
import wandb
from transformers import LogitsProcessor, LogitsProcessorList


class CustomLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: set[int]):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores[:, list(set(range(scores.shape[1])) - self.allowed_token_ids)] = float("-inf")
        return scores


class Message(TypedDict):
    role: str
    content: str


@dataclass
class GradedSample:
    question: str
    generated: str
    expected: str
    correct: bool
    is_first_hop_correct: bool | None = None
    prompt: str | None = None
    valid_cot: bool | None = None


async def grade(
    question: str,
    generated: str,
    expected: str | list[str],
    first_hop_reference_answer: str | list[str] | None = None,
    force_no_cot: bool = False,
    prompt: list[dict[str, str]] | None = None,
) -> GradedSample:
    expected = expected if isinstance(expected, list) else [expected]
    if first_hop_reference_answer is not None:
        first_hop_reference_answer = (
            first_hop_reference_answer
            if isinstance(first_hop_reference_answer, list)
            else [first_hop_reference_answer]
        )

    correct = False
    correct_generated_e3 = ""
    for exp in expected:
        if exp.lower() in generated.lower():
            correct = True
            correct_generated_e3 = exp.lower()
            break

    is_first_hop_correct = False
    correct_generated_e2 = ""
    if first_hop_reference_answer is not None and not force_no_cot:
        for reference_option in first_hop_reference_answer:
            # TODO: we might want model-grading here for more robustness
            if reference_option.lower() in generated.lower():
                is_first_hop_correct = True
                correct_generated_e2 = reference_option.lower()
                break
    return GradedSample(
        question=question,
        generated=generated,
        expected=str(expected),
        correct=correct,
        is_first_hop_correct=is_first_hop_correct,
        valid_cot=(
            correct
            and is_first_hop_correct
            and generated.lower().index(correct_generated_e2)
            < generated.lower().index(correct_generated_e3)  # name is before year
        ),
        prompt=str(prompt),
    )


def load_jsonl(file_path: str) -> list[dict[str, str]]:
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def add_few_shots(
    prompt: list[Message],
    few_shot_samples: list[dict[str, str]],
    rng: random.Random,
    max_few_shots: int | None = None,
) -> list[Message]:
    system_message = prompt[0]
    user_message = prompt[1]
    assert system_message["role"] == "system"
    assert user_message["role"] == "user"
    few_shots = []
    rng.shuffle(few_shot_samples)
    if max_few_shots:
        few_shot_samples = few_shot_samples[:max_few_shots]
    for example in few_shot_samples:
        messages = example["messages"]
        few_shots += [msg for msg in messages if msg["role"] != "system"]
    prompt = [system_message] + few_shots + [user_message]
    return prompt


def log_results_to_wandb(
    wandb_run,
    metric_name: str,
    results: dict[str, Any],
    samples: list[dict[str, Any]] = [],
    wandb_metrics: dict[str, Any] = {},
) -> None:
    results.update(wandb_metrics)
    wandb_run.log(results, commit=False)
    wandb_run.log(
        {
            f"samples_{metric_name.removeprefix('acc_')}": wandb.Table(
                dataframe=pd.DataFrame(samples)
            )
        }
    )


async def evaluate(
    model: str | transformers.PreTrainedModel,
    dataset_file: str,
    force_cot: bool = False,
    force_no_cot: bool = False,
    wandb_run_id: str | None = None,
    wandb_run: wandb.wandb_sdk.wandb_run.Run | None = None,
    wandb_project: str = "latent_reasoning",
    metric_name: str = "accuracy",
    few_shots_path: str | None = None,
    max_few_shots: int | None = None,
    subsample: int | None = None,
    wandb_metrics: dict[str, Any] = {},
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    do_sample: bool = False,
    **kwargs: Any,
):
    dataset = load_jsonl(dataset_file)
    rng = random.Random(42)
    if subsample:
        dataset = rng.choices(dataset, k=subsample)

    few_shot_samples = load_jsonl(few_shots_path) if few_shots_path else []
    # list of tuples:
    prompts: list[list[Message]] = [item["messages"] for item in dataset]
    questions = [item["question"] for item in dataset]
    first_hop_reference_answers = [item.get("answer_intermediate") for item in dataset]
    reference_answers = [item["answer"] for item in dataset]

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
        **kwargs,
    )
    pipeline.tokenizer.padding_side = "left"
    model_architecture = pipeline.model.config.architectures[0]
    if "llama" in model_architecture.lower():
        pipeline.tokenizer.pad_token = pipeline.tokenizer.decode(128_001)  # "<|end_of_text|>"
        pipeline.tokenizer.pad_token_id = 128_001
    elif "qwen" in model_architecture.lower():
        pipeline.tokenizer.pad_token = pipeline.tokenizer.decode(151_643)  # "<|end_of_text|>"
        pipeline.tokenizer.pad_token_id = 151_643

    if force_no_cot:
        # NOTE: used for fully-synthetic 2hop-no-CoT experiments since all no-CoT answers are single tokens
        allowed_tokens_ids = set([pipeline.tokenizer.eos_token_id, pipeline.tokenizer.pad_token_id])
        for answer in reference_answers:
            tokenized_answer = pipeline.tokenizer.encode(answer, add_special_tokens=False)
            assert len(tokenized_answer) == 1, "Answer must be a single token"
            allowed_tokens_ids.update(tokenized_answer)
    else:
        allowed_tokens_ids = set(pipeline.tokenizer.get_vocab().values())

    logits_processor = LogitsProcessorList([CustomLogitsProcessor(allowed_tokens_ids)])

    terminators = [pipeline.tokenizer.eos_token_id]

    # ensure last message is not assistant message
    for prompt in prompts:
        if prompt[-1]["role"] == "assistant":
            prompt.pop()

    if force_cot or force_no_cot:
        assert not (force_cot and force_no_cot), "Cannot force both COT and no COT"
        if force_cot:
            # We use few-shots to force strict CoT
            # For no-CoT, we may use few-shots or not, but not for OOD evaluation
            assert len(few_shot_samples) > 0, "Few-shot samples required for COT forcing"

        if len(few_shot_samples) > 0:
            prompts = [
                add_few_shots(prompt, few_shot_samples, rng, max_few_shots) for prompt in prompts
            ]

    time_before_inference = time.perf_counter()
    outputs = pipeline(
        prompts,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=do_sample,
        temperature=temperature,
        batch_size=8,
        logits_processor=logits_processor,
    )
    time_taken_inference = time.perf_counter() - time_before_inference
    print(f"Time taken for inference: {time_taken_inference:.2f}s")
    time_before_grading = time.perf_counter()

    tasks = []
    for question, model_output, first_hop_reference_answer, reference_answer in zip(
        questions, outputs, first_hop_reference_answers, reference_answers
    ):
        prompt = model_output[0]["generated_text"][:-1]
        model_answer = model_output[0]["generated_text"][-1]["content"]
        task_result = grade(
            question=question,
            generated=model_answer,
            expected=reference_answer,
            first_hop_reference_answer=first_hop_reference_answer,
            prompt=prompt,
        )
        tasks.append(task_result)

    graded_samples = await asyncio.gather(*tasks)
    time_taken_grading = time.perf_counter() - time_before_grading
    print(f"Time taken for grading: {time_taken_grading:.2f}s")

    is_corrects = [sample.correct for sample in graded_samples]
    is_first_hop_corrects = [sample.is_first_hop_correct for sample in graded_samples]
    valid_cots = [sample.valid_cot for sample in graded_samples]
    metrics = {}

    if not any(is_corrects):
        accuracy = 0
    else:
        accuracy = sum(is_corrects) / len(is_corrects)
    metrics[metric_name] = accuracy

    check_first_hop = any(first_hop_reference_answers)
    if check_first_hop:
        if not any(is_first_hop_corrects):
            accuracy_first_hop = 0
        else:
            first_hop_corrects_no_none = [x for x in is_first_hop_corrects if x is not None]
            accuracy_first_hop = sum(first_hop_corrects_no_none) / len(first_hop_corrects_no_none)
        metrics[metric_name + "_first_hop"] = accuracy_first_hop

    if force_no_cot:
        # Report "strict" accuracy: `is_correct` AND intermediate answer after answer year
        # FIXME: I've seen a case where a model does CoT but we count it as strict no-CoT,
        # in that case, the model answered with CoT but didn't mention the bridge entity.
        # Here is the table where I observed this: https://wandb.ai/apollo-evals/latent_reasoning/reports/Weave-samples_2hop_0shot-24-08-10-03-00-42---Vmlldzo4OTg2MzI1
        accuracy_strict = sum(
            [is_correct and not valid_cot for is_correct, valid_cot in zip(is_corrects, valid_cots)]
        ) / len(is_corrects)

        metrics[metric_name + "_strict"] = accuracy_strict

    print("Metrics:")
    for key, value in metrics.items():
        print(f" - {key}: {value:.2f}")

    my_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if my_rank == 0:
        wandb_run = wandb_run or wandb.init(id=wandb_run_id, project=wandb_project, resume=True)
        if wandb_run:
            log_results_to_wandb(
                wandb_run=wandb_run,
                metric_name=metric_name,
                results=metrics,
                samples=[sample.__dict__ for sample in graded_samples],
                wandb_metrics=wandb_metrics,
            )


if __name__ == "__main__":
    fire.Fire(evaluate)

"""
python evaluate.py models/$run_b_name datasets/artists/artists_2hop_test.jsonl --wandb_run_id $run_b_name --metric_name "acc_2hop_cot" --force_cot
"""

"""
python evaluate.py models/$run_b_name datasets/artists/artists_2hop_test.jsonl --wandb_run_id $run_b_name --metric_name "acc_2hop_0shot" --force_no_cot
"""

# TODO: test if I can get 0shot 2-hop without logit bias -> 20 fewshots not enough
# TODO: also logit bias may be fine for 2-hop for cities.
