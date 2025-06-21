# %%
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import wandb.apis.public.runs


# Experiment 2: Same document & In-context

def fetch_runs_for_setting(
    project_name: str,
    config_path: str,
    # tag: str = "arxiv",
    tag: str = "arxiv",
) -> List["wandb.apis.public.runs.Run"]:
    api = wandb.Api()
    runs = api.runs(
        path=project_name,
        filters={
            "tags": {"$in": [tag]},
            "config.experiment_config_path": config_path,
            "state": "finished",
        },
    )
    return runs

def create_figure1_three_settings_plot():
    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(figsize=(6, 6))

    # Data points
    settings = [
        "In-context",
        "Same\n docs",
        "Separate\n docs",
    ]

    # Get averages and errors from wandb for the multi-run settings
    # Fetch runs for "both hops in same doc"
    same_doc_runs = fetch_runs_for_setting(
        project_name="sita/latent_reasoning",
        config_path="experiments/arxiv/both_hops_samedoc.yaml",
    )
    same_doc_accs = []
    for run in same_doc_runs:
        history = run.scan_history()
        data = pd.DataFrame(history)
        acc = data["acc_2hop_0shot_strict"].dropna().iloc[-1]
        same_doc_accs.append(acc * 100)
    same_doc_avg = np.mean(same_doc_accs)
    same_doc_err = np.std(same_doc_accs) / np.sqrt(len(same_doc_accs))  # Standard error

    # Fetch runs for "two hop no cot"
    separate_doc_runs = fetch_runs_for_setting(
        project_name="sita/latent_reasoning",
        config_path="experiments/fully_synthetic/configs/no_cot_and_cot.yaml",
    )
    separate_doc_accs = []
    for run in separate_doc_runs:
        history = run.scan_history()
        data = pd.DataFrame(history)
        acc = (
            data["acc_2hop_0shot_strict"].dropna().iloc[-1]
            if not data["acc_2hop_0shot_strict"].dropna().empty
            else 0
        )
        separate_doc_accs.append(acc * 100)
    separate_doc_avg = np.mean(separate_doc_accs)
    separate_doc_err = np.std(separate_doc_accs) / np.sqrt(len(separate_doc_accs))  # Standard error

    # In-context values
    in_context_accs = [
        67.08,
        63.37,
        67.49,
    ]  # inspect_ai runs aren't logged to W&B; pasted these from command-line output
    in_context_avg = np.mean(in_context_accs)
    in_context_err = np.std(in_context_accs) / np.sqrt(len(in_context_accs))  # Standard error

    values = [in_context_avg, same_doc_avg, separate_doc_avg]
    errors = [in_context_err, same_doc_err, separate_doc_err]

    colors = [
        "#88C7C3",
        "#BF635F",
        "#F2A93B",
    ]
    ax.bar(settings, values, yerr=errors, capsize=5, color=colors)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    plt.xticks(fontsize=22)
    sns.despine()
    plt.tight_layout()

    return fig, ax

fig, ax = create_figure1_three_settings_plot()
fig.savefig("figure1_three_settings.pdf", dpi=300)
plt.show()
