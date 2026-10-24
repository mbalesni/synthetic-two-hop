"""Offline compatibility checks.

Run with `uv run --locked --all-extras python scripts/security_smoke.py`.
"""

import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

    for module in (
        "latent_reasoning.train",
        "latent_reasoning.trainer",
        "latent_reasoning.evaluate",
        "inspect_ai",
        "langchain_anthropic",
        "langchain_openai",
        "langgraph_sdk",
        "litellm",
        "aiohttp",
        "certifi",
        "tornado",
        "urllib3",
        "multidict",
        "s3fs",
    ):
        importlib.import_module(module)

    import torch
    from accelerate import Accelerator
    from filelock import FileLock
    from git import Repo
    from transformers import LlamaConfig, LlamaForCausalLM

    from datasets import Dataset, load_from_disk

    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    accelerator = Accelerator(cpu=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model, optimizer = accelerator.prepare(model, optimizer)
    inputs = torch.randint(0, 32, (2, 8))
    before = next(model.parameters()).detach().clone()
    loss = model(input_ids=inputs, labels=inputs).loss
    assert torch.isfinite(loss)
    accelerator.backward(loss)
    optimizer.step()
    assert not torch.equal(before, next(model.parameters()).detach())

    with tempfile.TemporaryDirectory(prefix="security-smoke-") as directory:
        root = Path(directory)
        with FileLock(root / "test.lock", timeout=2):
            dataset = Dataset.from_dict({"text": ["one", "two"]})
            dataset.save_to_disk(root / "dataset")
            assert load_from_disk(root / "dataset")[:]["text"] == ["one", "two"]
        repo = Repo.init(root / "repo")
        (root / "repo" / "sample.txt").write_text("test\n")
        repo.index.add(["sample.txt"])
        assert repo.index.entries
        repo.close()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "virtualenv",
                "--no-download",
                "--no-periodic-update",
                "--app-data",
                str(root / "cache"),
                str(root / "venv"),
            ],
            check=True,
            env={**os.environ, "VIRTUALENV_OVERRIDE_APP_DATA": str(root / "cache")},
            timeout=60,
        )
        subprocess.run(
            [str(root / "venv" / "bin" / "python"), "-m", "pip", "--version"], check=True
        )

    print("Imports, CPU training, dataset persistence, file locking, Git, and virtualenv passed.")


if __name__ == "__main__":
    main()
