import asyncio
import gc
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any

import debugpy
import torch
import torch.distributed
import torch.utils.data
import wandb
import yaml
from accelerate.state import PartialState
from devtools import pprint
from rich.logging import RichHandler
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from transformers import AutoConfig, AutoTokenizer, TrainerCallback, TrainingArguments
from trl import ModelConfig, get_peft_config
from trl.commands.cli_utils import SftScriptArguments, TrlParser, init_zero_verbose

from datasets import concatenate_datasets, load_dataset
from latent_reasoning.common import AuxLossType, merge_system_message
from latent_reasoning.evaluate import evaluate
from latent_reasoning.trainer import CustomDataCollator, CustomTrainer


def attach_debugger(port=5678):
    debugpy.listen(port)
    print(f"Waiting for debugger on port {port}...")

    debugpy.wait_for_client()
    print(f"Debugger attached on port {port}")


@dataclass
class CustomArgs:
    experiment_config_path: str
    aux_loss: bool | None = None
    aux_loss_coef: float | None = None
    aux_loss_target_layer: int | None = None
    aux_loss_type: AuxLossType | None = None
    layer_range: str | None = None  # e.g. "0-12"
    wandb_project: str = "latent_reasoning"

    def __post_init__(self):
        assert os.path.exists(
            self.experiment_config_path
        ), f"Config file {self.experiment_config_path} not found"
        with open(self.experiment_config_path, "r") as file:
            experiment_config = yaml.safe_load(file)
            for key, value in experiment_config.items():
                if getattr(self, key, None) is None:
                    setattr(self, key, value)

    def __getitem__(self, key: str, **kwargs: Any):
        return getattr(self, key, **kwargs)


class EvaluationCallback(TrainerCallback):
    def __init__(self, evaluation_config: list[dict[str, Any]]):
        self.evaluations = evaluation_config

    def evaluate_with_sampling(self, model, tokenizer, state, **kwargs):
        print(f"{state.global_step=} {state.epoch=} Evaluating...")

        async def evaluation_loop():
            with FullyShardedDataParallel.summon_full_params(model, recurse=False):
                for evaluation_config in self.evaluations:
                    print(f"Evaluating {evaluation_config['metric_name']=}")

                    await evaluate(
                        model=model.module
                        if isinstance(model, FullyShardedDataParallel)
                        else model,  # type: ignore
                        tokenizer=tokenizer,
                        wandb_run=wandb.run,
                        wandb_metrics={
                            "train/global_step": state.global_step,
                            "train/epoch": state.epoch,
                        },
                        **evaluation_config,
                    )
                    print(f"Done with {evaluation_config['metric_name']=}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(evaluation_loop())
        loop.close()

        # clear cuda memory
        torch.cuda.empty_cache()
        gc.collect()

    def on_train_begin(self, args, state, control, model, tokenizer, **kwargs):
        self.evaluate_with_sampling(model, tokenizer, state, **kwargs)
        return control

    def on_epoch_end(self, args, state, control, model, tokenizer, **kwargs):
        self.evaluate_with_sampling(model, tokenizer, state, **kwargs)
        return control


def run_hf_finetuning(
    training_args: TrainingArguments,
    custom_args: CustomArgs,
    model_config: ModelConfig,
) -> None:
    # clear cuda memory
    torch.cuda.empty_cache()

    model_name = model_config.model_name_or_path or custom_args["model_name_or_path"]
    custom_args.model_name_or_path = model_name
    model_config.model_name_or_path = model_name
    if PartialState().is_local_main_process:
        print("Experiment config:")
        pprint(custom_args)

    my_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print(f"Hi, I'm train.py, running on node {os.uname().nodename} with rank {my_rank}.")

    train_datasets = {
        dataset["name"]: load_dataset(
            "json",
            data_files=dataset["dataset_file"],
            cache_dir=None,
        )["train"]  # type: ignore
        for dataset in custom_args["train_datasets"]
    }
    for dataset_name, dataset in train_datasets.items():
        print(f"Loaded {dataset_name=} with {len(dataset)=}")

    train_dataset = concatenate_datasets(list(train_datasets.values()))  # type: ignore
    print(f"Concatenated datasets into one with {len(train_dataset)=}")

    eval_dataset = load_dataset(
        "json",
        data_files={d["name"]: d["dataset_file"] for d in custom_args["eval_datasets"]},
        cache_dir=None,
    )
    if "gemma" in model_name.lower():
        train_dataset = train_dataset.map(merge_system_message)
        eval_dataset = eval_dataset.map(merge_system_message)
    for partition in eval_dataset.keys():
        config = [config for config in custom_args["eval_datasets"] if config["name"] == partition][
            0
        ]
        n_samples = config.get("subsample", None)
        if isinstance(n_samples, int):
            eval_dataset[partition] = eval_dataset[partition].shuffle().select(range(n_samples))
            assert len(eval_dataset[partition]) == n_samples

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(model_name)
    model_architecture = config.architectures[0]
    if "llama" in model_architecture.lower():
        tokenizer.pad_token = tokenizer.decode(128_001)  # "<|end_of_text|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "qwen" in model_architecture.lower():
        tokenizer.pad_token = tokenizer.decode(151_643)  # "<|end_of_text|>"
        response_template = "<|im_start|>assistant\n"
    elif "gemma" in model_architecture.lower():
        tokenizer.pad_token = tokenizer.decode(0)  # "<pad>"
        response_template = "<start_of_turn>model\n"
    elif "olmo" in model_architecture.lower():
        response_template = "<|assistant|>\n"
    else:
        logging.warning(f"Unknown model {model_name}")

    collator = CustomDataCollator(
        response_template, tokenizer=tokenizer
    )  # this makes sure prompt is masked out

    job_info = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }

    if my_rank == 0:
        run = wandb.init(
            project=custom_args.wandb_project,
            config={
                **training_args.__dict__,
                **model_config.__dict__,
                **custom_args.__dict__,
                **job_info,
            },
        )

    trainer = CustomTrainer(
        model=model_name,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            EvaluationCallback(evaluation_config=custom_args["evaluations"]),
        ],
        data_collator=collator,
        dataset_kwargs=dict(add_special_tokens=False),
        peft_config=get_peft_config(model_config),
        aux_loss=custom_args["aux_loss"],
        aux_loss_coef=custom_args["aux_loss_coef"],
        aux_loss_target_layer=custom_args["aux_loss_target_layer"],
        aux_loss_type=custom_args["aux_loss_type"],
    )

    # Freeze all layers except the ones in layer_range
    layer_range = custom_args.layer_range
    if layer_range:
        # Freeze all layers
        for param in trainer.model.parameters():
            param.requires_grad = False

        start, end = layer_range.split("-")
        start, end = int(start), int(end)
        for layer_id in range(start, end):
            for param in trainer.model.model.layers[layer_id].parameters():
                param.requires_grad = True

        print(f"Froze all layers except those in range {layer_range}")
    else:
        print("No layer range specified. All layers will be trained.")

    for name, param in trainer.model.named_parameters():
        print(f"{name=} {param.requires_grad=}")

    trainer.train()

    print("Training completed!")
    # print(f"Training completed! Saving the model to {training_args.output_dir}")
    # trainer.save_model(training_args.output_dir)
    if my_rank == 0:
        run.finish()

    # clear cuda memory
    torch.cuda.empty_cache()


if __name__ == "__main__":
    init_zero_verbose()
    logging.basicConfig(
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
        level=logging.INFO,
    )
    parser = TrlParser((SftScriptArguments, TrainingArguments, ModelConfig, CustomArgs))
    args, training_args, model_config, custom_args = parser.parse_args_and_config()
    os.environ["WANDB_PROJECT"] = custom_args.wandb_project
    print(f"Hostname: {os.uname().nodename}, SLURM Job: {os.environ.get('SLURM_JOB_ID')}")
    run_hf_finetuning(
        training_args=training_args,
        custom_args=custom_args,
        model_config=model_config,
    )

# Main model
"""
accelerate launch \
--num_processes 2 \
--config_file fsdp_accelerate_config.yaml \
--main_process_port 29503 train.py \
--config trl_config.yaml \
--output_dir models/test-llama-3-8b \
--run_name "first tests of synthetic spouses" \
--experiment_config experiments/ft_synthetic_spouses/atomic.yaml \
--model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct"
--use_peft_lora --lora_r=1 --lora_alpha=16
"""

# Auxiliary loss [type: logit, layer: 10, coef: 1.0]
"""
accelerate launch \
--num_processes 2 \
--config_file fsdp_accelerate_config.yaml \
--main_process_port 29503 train.py \
--config trl_config.yaml \
--output_dir models/test-llama-3-8b \
--run_name "first tests of synthetic spouses" \
--experiment_config experiments/ft_synthetic_spouses/atomic+2hop_cot_nocot_aux.yaml \
--model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct"
"""


# DEBUG: tiny model, 1 GPU, no W&B (NOTE: doesn't work for aux loss)
"""
WANDB_MODE=disabled accelerate launch \
--num_processes 1 \
--config_file fsdp_accelerate_config.yaml \
--main_process_port 29503 train.py \
--config trl_config.yaml \
--output_dir models/debug \
--run_name "debug" \
--experiment_config experiments/ft_synthetic_spouses/atomic+2hop_cot_nocot.yaml \
--model_name_or_path "sshleifer/tiny-gpt2"
"""

"""
accelerate launch \
--num_processes 2 \
--config_file fsdp_accelerate_config.yaml \
--main_process_port 29503 train.py \
--config trl_config.yaml \
--output_dir models/debug \
--run_name "debug" \
--experiment_config experiments/arxiv/data_mixture/no_cot_and_cot.yaml \
--model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct"
"""
# --model_name_or_path "Qwen/Qwen2.5-7B-Instruct"
