import os
from typing import Any, Union

import torch
import torch.nn.functional as F
import transformers
import wandb
from transformers import PreTrainedModel
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from torch.utils.data import Dataset
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from latent_reasoning.common import AuxLossType


class TomnikContainer:
    def __init__(self, value: Any, **kwargs):
        self.payload = value

    def to(self, **kwargs):
        return self.payload


class CustomDataCollator(DataCollatorForCompletionOnlyLM):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        def custom_pad_without_fast_tokenizer_warning(tokenizer, examples, **pad_kwargs):
            pad_kwargs.pop("return_tensors", None)
            # remove field `messages` form all exampleks
            for example in examples:
                example.pop("messages", None)

            batch = pad_without_fast_tokenizer_warning(
                # i removed return_tensors="pt",
                tokenizer,
                examples,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            # convert input_ids and attention_mask to torch tensor
            batch["input_ids"] = torch.tensor(batch["input_ids"])
            batch["attention_mask"] = torch.tensor(batch["attention_mask"])
            if "question" in batch:
                batch["question"] = TomnikContainer(batch["question"])
            if "answer_intermediate" in batch:
                batch["answer_intermediate"] = TomnikContainer(batch["answer_intermediate"])
            if "answer" in batch:
                batch["answer"] = TomnikContainer(batch["answer"])
            if "auxiliary_loss_prefix" in batch:
                batch["auxiliary_loss_prefix"] = TomnikContainer(batch["auxiliary_loss_prefix"])
            return batch

        transformers.data.data_collator.pad_without_fast_tokenizer_warning = (
            custom_pad_without_fast_tokenizer_warning
        )


def find_subsequence_end(bigger: torch.Tensor, smaller: torch.Tensor) -> int:
    n, m = bigger.size(0), smaller.size(0)

    if m > n:
        return -1  # Subsequence is longer than the main sequence

    for i in range(n - m + 1):
        if torch.all(bigger[i : i + m] == smaller):
            return i + m - 1

    return -1  # Subsequence not found


class CustomTrainer(SFTTrainer):
    def __init__(
        self,
        aux_loss: bool = False,
        aux_loss_coef: float = 1.0,
        aux_loss_target_layer: int = 10,
        aux_loss_source_layer: int = 10,
        aux_loss_type: AuxLossType | None = None,
        aux_loss_collected_activations_path: str | None = None,
        *args,
        **kwargs,
    ):
        if aux_loss:
            kwargs["args"].remove_unused_columns = False

        super().__init__(*args, **kwargs)
        self.aux_loss = aux_loss
        self.aux_loss_coef = aux_loss_coef
        self.aux_loss_target_layer = aux_loss_target_layer
        self.aux_loss_source_layer = aux_loss_source_layer
        self.aux_loss_type = aux_loss_type
        self.aux_loss_collected_activations_path = aux_loss_collected_activations_path


    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Any,
        return_outputs=False,
    ):
        assert (
            self.args.past_index == -1 and self.label_smoother is None
        ), "Not supported by CustomTrainer but who cares 🤷🏻"
        assert self.tokenizer is not None, "Tokenizer is not set"

        inputs.pop("question", None)
        inputs.pop("answer", None)
        answer_intermediates = inputs.pop("answer_intermediate", None)
        auxiliary_loss_prefixes = inputs.pop("auxiliary_loss_prefix", None)

        outputs = model(**inputs, output_hidden_states=True)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        stage = "train" if model.training else "eval"
        custom_logs = {
            f"{stage}/original_loss": loss.item(),
            "train/global_step": self.state.global_step,
        }
        if self.aux_loss:
            aux_examples_indices, e2_description_end_token_positions = [], []
            for example_idx, (question_tokenized, auxiliary_loss_prefix) in enumerate(
                zip(inputs.input_ids, auxiliary_loss_prefixes, strict=True)
            ):
                if auxiliary_loss_prefix != "":
                    auxiliary_loss_prefix_tokenized = (
                        self.tokenizer(
                            auxiliary_loss_prefix, return_tensors="pt", add_special_tokens=False
                        )
                        .input_ids[0]
                        .to(question_tokenized.device)
                    )
                    # find the last token position where auxiliary_loss_prefix_tokenized is a subsequence of question_tokenized
                    last_token_idx = find_subsequence_end(
                        question_tokenized, auxiliary_loss_prefix_tokenized
                    )
                    assert last_token_idx != -1, "Subsequence not found"
                    assert last_token_idx < question_tokenized.size(
                        0
                    ), "Subsequence is longer than the main sequence"

                    e2_description_end_token_positions.append(last_token_idx)
                    aux_examples_indices.append(example_idx)

            if len(aux_examples_indices) > 0:
                auxiliary_labels = [
                    ans for i, ans in enumerate(answer_intermediates) if i in aux_examples_indices
                ]
                aux_loss = self.compute_auxiliary_loss(
                    model=model,
                    hidden_states=outputs.hidden_states,
                    examples_indices=aux_examples_indices,
                    position_indices=e2_description_end_token_positions,
                    labels=auxiliary_labels,
                )
                if stage == "eval":
                    current_eval_dataset_name = self.state.current_eval_dataset_name
                    custom_logs[f"{stage}/{current_eval_dataset_name}/aux_loss"] = aux_loss.item()
                    custom_logs[f"{stage}/{current_eval_dataset_name}/orig_loss"] = loss.item()
                else:
                    custom_logs[f"{stage}/aux_loss"] = aux_loss.item()
                    custom_logs[f"{stage}/orig_loss"] = loss.item()
                loss = self.aux_loss_coef * aux_loss + loss
        if self.is_world_process_zero():
            wandb.log(custom_logs, commit=False)
        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Union[Dataset, dict[str, Dataset]] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        # we need to set this before calling super().evaluate such that compute_loss can use it
        # i'm sorry
        self.state.current_eval_dataset_name = eval_dataset
        return super().evaluate(
            eval_dataset, ignore_keys, metric_key_prefix
        )

    def compute_auxiliary_loss(
        self,
        model,
        hidden_states,
        examples_indices: list[int],
        position_indices: list[int],
        labels: list[str],
    ):
        activations_to_apply_loss_on = hidden_states[self.aux_loss_target_layer][
            examples_indices, position_indices, :
        ]  # expected: [batch_size, hidden_size]
        labels_tokenized = (
            self.tokenizer(labels, return_tensors="pt", add_special_tokens=False)
            .input_ids.to(activations_to_apply_loss_on.device)
            .squeeze(dim=1)
        )  # expected: [batch_size]

        activations_to_apply_loss_on = activations_to_apply_loss_on.to(
            model.lm_head.weight.dtype
        )  # cast to float16
        if self.aux_loss_type == "logit":
            logits = self.logit_lens_no_grad(model, activations_to_apply_loss_on)
            logits = logits.float()  # cast to float32
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels_tokenized)
        elif self.aux_loss_type == "embed_cosine":
            bridge_entity_embeddings = model.model.embed_tokens(labels_tokenized)
            bridge_entity_embeddings = (
                bridge_entity_embeddings.float()
            )  # TODO: try replacing with activations_to_apply_loss_on.dtype
            loss_fn = torch.nn.CosineSimilarity()
            loss = -1 * loss_fn(activations_to_apply_loss_on, bridge_entity_embeddings).mean()
        elif self.aux_loss_type == "embed_mse":
            bridge_entity_embeddings = model.model.embed_tokens(labels_tokenized)
            bridge_entity_embeddings = bridge_entity_embeddings.to(
                activations_to_apply_loss_on.dtype
            )
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(activations_to_apply_loss_on, bridge_entity_embeddings)
        else:
            raise ValueError(f"Unknown aux loss type: {self.aux_loss_type}")

        return loss

    def logit_lens_no_grad(self, model: PreTrainedModel, input: torch.Tensor):
        # Step 1: norm(input)
        input_dtype = input.dtype
        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + model.model.norm.variance_epsilon)
        normed = model.model.norm.weight.detach() * input.to(input_dtype)
        # Step 2: lm_head(normed)
        lm_headed = F.linear(normed, model.lm_head.weight.detach())
        return lm_headed
