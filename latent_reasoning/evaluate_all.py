import fire
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_reasoning.evaluate import evaluate


async def evaluate_all(
    model_id: str,
    wandb_run_id: str | None = None,
    evaluation_config: str = "experiment_config.yaml",
):
    # load the model with transformers
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, padding_side="left"
    )

    # load yaml config
    with open(evaluation_config, "r") as file:
        config = yaml.safe_load(file)

    for eval_config in config["evaluations"]:
        assert isinstance(eval_config, dict)

        await evaluate(
            model=model,
            tokenizer=tokenizer,
            wandb_run_id=wandb_run_id,
            **eval_config,
        )


if __name__ == "__main__":
    fire.Fire(evaluate_all)

"""
python evaluate_all.py models/$run_b_name [--wandb_run_id $run_b_name]
"""
