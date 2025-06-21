# %%


import logging
import os
import re
import sys
from typing import Literal

import fire
import pandas as pd
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer with activation hooks setup."""
    logger.info(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir="/data/huggingface/",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_prompts(entities: list[str]) -> list[list[dict[str, str]]]:
    """Create prompts for getting entity representations in batch."""
    return [[{"role": "user", "content": f"Tell me about {entity}"}] for entity in entities]


class ActivationHookManager:
    def __init__(self, model: AutoModelForCausalLM):
        self.model = model
        self.hooks = []
        self.activations = {}

    def register_hooks(self):
        """Register hooks for all transformer blocks."""

        def hook_fn(layer_name):
            def forward_hook(module, input, output):
                self.activations[layer_name] = (
                    output[0].detach().cpu()
                )  # [batch, seq_len, hidden_dim]

            return forward_hook

        # Register a hook for each transformer block
        for name, module in self.model.named_modules():
            if "layers" in name and isinstance(
                module, transformers.models.llama.modeling_llama.LlamaDecoderLayer
            ):
                layer_hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(layer_hook)

    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_batch_activations(self) -> dict[str, torch.Tensor]:
        """Get activations for each item in the batch."""
        batch_activations = {}
        for layer_name, acts in self.activations.items():
            batch_activations[layer_name] = []
            # acts shape: [batch, seq_len, hidden_dim]
            batch_size = acts.shape[0]
            for i in range(batch_size):
                batch_activations[layer_name].append(acts[i, -1].cpu())
        return batch_activations


def entity_to_filename(entity: str) -> str:
    """Convert arbitrary string to filename safely."""
    return re.sub(r"[^\w\-_]", "_", entity)


def collect_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    entities: list[str],
    save_dir: str,
    batch_size: int = 8,
    ignore_eot_token: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Collect and save activations for a list of entities.
    Returns a dictionary mapping entity names to their activations.
    """
    hook_manager = ActivationHookManager(model)
    hook_manager.register_hooks()
    os.makedirs(save_dir, exist_ok=True)

    results = {}

    # Process entities in batches
    progress = tqdm(total=len(entities), desc="Collecting entity activations", file=sys.stdout)
    for i in range(0, len(entities), batch_size):
        batch_entities = entities[i : i + batch_size]
        batch_to_process = []

        # Filter out already processed entities
        for entity in batch_entities:
            entity_filename = entity_to_filename(entity)
            if not os.path.exists(f"{save_dir}/{entity_filename}.pt"):
                batch_to_process.append(entity)

        progress.update(len(batch_entities))
        if not batch_to_process:
            continue

        # Process the batch
        prompts = create_prompts(batch_to_process)
        input_ids = tokenizer.apply_chat_template(
            prompts, add_generation_prompt=False, padding=True, return_tensors="pt"
        ).to(model.device)

        if ignore_eot_token:
            input_ids = input_ids[:, :-1]

        with torch.no_grad():
            _ = model(input_ids)

        # Get activations for each entity in the batch
        batch_activations = hook_manager.get_batch_activations()

        # Save activations for each entity
        for idx, entity in enumerate(batch_to_process):
            entity_filename = entity_to_filename(entity)
            entity_activations = {
                layer_name: acts[idx].clone() for layer_name, acts in batch_activations.items()
            }

            torch.save(entity_activations, f"{save_dir}/{entity_filename}.pt")
            results[entity] = entity_activations

        hook_manager.clear_activations()

    hook_manager.remove_hooks()
    return results


def main(
    model_name: str,  # "meta-llama/Meta-Llama-3.1-70B-Instruct",
    batch_size: int = 8,
    dataset: Literal["hopping_too_late", "synthetic_spouses"] = "hopping_too_late",
):
    model, tokenizer = load_model(model_name)
    if dataset == "hopping_too_late":
        triplets_df = pd.read_csv("two_hop_results_raw_hopping_too_late_pivot.csv")
        logger.info(f"Loaded {len(triplets_df)} valid triplets")

        # Collect activations:
        model_name_pathsafe = model_name.replace("/", "_")
        entities = sorted(
            list(
                set(
                    triplets_df["e1_label"].tolist()
                    + triplets_df["e2_label"].tolist()
                    + triplets_df["e3_label"].tolist()
                )
            )
        )
    elif dataset == "synthetic_spouses":
        raise NotImplementedError("Synthetic spouses not implemented yet")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    collect_activations(
        model,
        tokenizer,
        entities,
        save_dir=os.path.join(script_dir, "activations", model_name_pathsafe),
        batch_size=batch_size,
    )


if __name__ == "__main__":
    fire.Fire(main)

"""
python latent_reasoning/interp/collect_activations.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"
"""
