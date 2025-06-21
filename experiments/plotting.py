# %%
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

# Constants
WANDB_PROJECT = "sita/latent_reasoning"

# Define model colors
MODEL_COLORS = {
    "LLaMA-3-8B-Instruct": "#F2A93B",  # for matplotlib, hex color format is:
    "Qwen2.5-7B-Instruct": "#C0DB84",
    "GPT-4o-mini": "#C8D1F4",
    "GPT-4o": "#8095E5",
}

FIGURE_DIR = Path("../")  # noqa: F821

# Add these constants after the existing MODEL_COLORS definition
EXPERIMENT4_1_SETTINGS = [
    {
        "label": "Baseline",
        "color": "#F2A93B",
        "runs": None,  # Will use experiment1 data
    },
    {
        "label": "Staged, all layers",
        "color": "#88C7C3",
        "runs": [
            "sita/latent_reasoning/oqcohwcq",
            "sita/latent_reasoning/krjmh4bg",
            "sita/latent_reasoning/larhilh2",
        ],
    },
    {
        "label": "Staged, layer-selective",
        "color": "#8095E5",
        "runs": [
            "sita/latent_reasoning/smwdxxt6",
            "sita/latent_reasoning/03bhad2w",
            "sita/latent_reasoning/35vufgnk",
        ],
    },
]


def fetch_runs(
    config_path: str, tag: str = "arxiv", extra_filters: Dict[str, Any] | None = None
) -> List[wandb.apis.public.Run]:
    """Fetch W&B runs for a specific configuration"""
    api = wandb.Api()
    runs = api.runs(
        path=WANDB_PROJECT,
        filters={
            "tags": {"$in": [tag]},
            "config.experiment_config_path": config_path,
            "state": "finished",
            **(extra_filters or {}),
        },
    )
    return runs


def fetch_runs_for_model(
    config_path: str,
    architecture: str,
    tag: str = "arxiv",
    extra_filters: Dict[str, Any] | None = None,
) -> List[wandb.apis.public.Run]:
    """Fetch W&B runs for a specific configuration and model architecture"""
    runs = fetch_runs(config_path, tag, extra_filters)
    # Filter runs based on architecture
    filtered_runs = [
        run
        for run in runs
        if isinstance(run.config.get("architectures"), list)
        and any(architecture.lower() in arch.lower() for arch in run.config["architectures"])
    ]
    return filtered_runs


def process_runs(runs: List[wandb.apis.public.Run]) -> Dict[str, List[float]]:
    """Extract metrics from runs"""
    metrics = {
        "acc_a": [],
        "acc_b": [],
        "acc_2hop_cot": [],
        "acc_2hop_0shot_strict": [],
    }

    for run in runs:
        history = run.scan_history()
        data = pd.DataFrame(history)
        for metric in metrics:
            value = data[metric].dropna().iloc[-1] if not data[metric].dropna().empty else 0
            metrics[metric].append(value)

    return metrics


def format_mean_std(values: List[float]) -> str:
    """Format mean ± std as string"""
    if not values:
        return "—"
    return f"{np.mean(values):.3f} ± {np.std(values):.3f}"


def fetch_experiment1_data():
    """Fetch and process data for Experiment 1"""
    config_path = "experiments/fully_synthetic/configs/no_cot_and_cot.yaml"
    llama_metrics = process_runs(fetch_runs_for_model(config_path, "LLaMAForCausalLM"))
    qwen_metrics = process_runs(fetch_runs_for_model(config_path, "Qwen2ForCausalLM"))

    gpt_4o_mini_metrics = {
        "acc_a": [1.0, 1.0, 1.0],
        "acc_b": [1.0, 1.0, 1.0],
        "acc_2hop_cot": [0.733, 0.724, 0.984],
        "acc_2hop_0shot_strict": [0.008, 0.000, 0.012],
    }

    gpt_4o_metrics = {
        "acc_a": [1.0, 1.0, 1.0],
        "acc_b": [1.0, 1.0, 1.0],
        "acc_2hop_cot": [0.34, 0.69, 0.52],
        "acc_2hop_0shot_strict": [0.00, 0.00, 0.00],
    }

    return {
        "LLaMA-3-8B-Instruct": llama_metrics,
        "Qwen2.5-7B-Instruct": qwen_metrics,
        "GPT-4o-mini": gpt_4o_mini_metrics,
        "GPT-4o": gpt_4o_metrics,
    }


def create_experiment1_accuracy_plot(experiment_data):
    """Create bar plot for Experiment 1 results"""
    # Prepare data for plotting
    metrics_map = {
        "1st hop": "acc_a",
        "2nd hop": "acc_b",
        "Two-hop\nwith CoT": "acc_2hop_cot",
        "Two-hop\nwithout CoT": "acc_2hop_0shot_strict",
    }

    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot bars for each metric
    x = np.arange(len(metrics_map))
    width = 0.2  # Width of bars

    for i, (model, metrics) in enumerate(experiment_data.items()):
        values = []
        errors = []
        for metric_key in metrics_map.values():
            metric_values = metrics[metric_key]
            mean_value = np.mean(metric_values) * 100
            # Calculate standard error (if more than one value exists)
            if len(metric_values) > 1:
                std_error = (np.std(metric_values, ddof=1) / np.sqrt(len(metric_values))) * 100
            else:
                std_error = 0
            values.append(mean_value)
            errors.append(std_error)

        plt.bar(
            x + i * width,
            values,
            width,
            yerr=errors,
            capsize=5,  # Add caps to error bars
            label=model,
            color=MODEL_COLORS[model],
        )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    plt.xticks(x + width * 1.5, metrics_map.keys(), fontsize=18)
    sns.despine()
    plt.tight_layout()
    plt.show()

    return fig, ax


def create_experiment1_combined_loss_plot(llama_runs, qwen_runs):
    """Create combined loss plot showing two-hop no-CoT losses for both models"""
    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(figsize=(9, 6))

    # Define metrics and styling
    loss_columns = ["eval/2hop_no_cot_loss", "eval/2hop_no_cot_shuffled_loss"]
    styles = ["-", ":"]  # Solid for normal, dotted for random

    # Use model colors from accuracy plot
    llama_color = MODEL_COLORS["LLaMA-3-8B-Instruct"]
    qwen_color = MODEL_COLORS["Qwen2.5-7B-Instruct"]

    # Plot LLaMA losses
    for col, style in zip(loss_columns, styles):
        all_data = []
        for run in llama_runs:
            history = run.scan_history()
            df = pd.DataFrame(history)
            if not df.empty and col in df.columns:
                df = df[["train/epoch", col]].dropna()
                all_data.append(df)

        if all_data:
            combined_data = pd.concat(all_data, axis=0)
            label = "LLaMA" if style == "-" else "LLaMA (random)"
            sns.lineplot(
                data=combined_data,
                x="train/epoch",
                y=col,
                label=label,
                color=llama_color,
                linestyle=style,
                alpha=0.55 if style == "-" else 1.0,
                linewidth=5,
                errorbar="se",
                ax=ax,
                legend=False,
            )

    # Plot Qwen losses
    for col, style in zip(loss_columns, styles):
        all_data = []
        for run in qwen_runs:
            history = run.scan_history()
            df = pd.DataFrame(history)
            if not df.empty and col in df.columns:
                df = df[["train/epoch", col]].dropna()
                all_data.append(df)

        if all_data:
            combined_data = pd.concat(all_data, axis=0)
            label = "Qwen" if style == "-" else "Qwen (random)"
            sns.lineplot(
                data=combined_data,
                x="train/epoch",
                y=col,
                label=label,
                color=qwen_color,
                linestyle=style,
                alpha=0.55 if style == "-" else 1.0,
                linewidth=5,
                errorbar="se",
                ax=ax,
                legend=False,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Loss")
    ax.set_xlim(0, 1)
    plt.xticks(fontsize=20)
    sns.despine()
    plt.tight_layout()

    return fig, ax


def create_experiment4_1_accuracy_plot(experiment1_data, main_font_size=22):
    """Create accuracy plot for Experiment 4.1 results"""
    plt.rcParams.update({"font.size": main_font_size})
    fig, ax = plt.subplots(figsize=(9, 6))

    metrics_map = {
        "1st hop": "acc_a",
        "2nd hop": "acc_b",
        "Two-hop\nwith CoT": "acc_2hop_cot",
        "Two-hop\nwithout CoT": "acc_2hop_0shot_strict",
    }

    x = np.arange(len(metrics_map))
    width = 0.25  # Width of bars

    api = wandb.Api()

    # Plot bars for each setting
    for i, setting in enumerate(EXPERIMENT4_1_SETTINGS):
        values = []
        errors = []

        if setting["runs"] is None:
            # Use experiment1 data for baseline
            metrics = experiment1_data["LLaMA-3-8B-Instruct"]
            for metric_key in metrics_map.values():
                metric_values = metrics[metric_key]
                values.append(np.mean(metric_values) * 100)
                errors.append((np.std(metric_values, ddof=1) / np.sqrt(len(metric_values))) * 100)
        else:
            # Fetch data from W&B runs
            all_metrics = {metric: [] for metric in metrics_map.values()}
            for run_path in setting["runs"]:
                run = api.run(run_path)
                history = run.scan_history()
                data = pd.DataFrame(history)

                for metric_key in metrics_map.values():
                    value = (
                        data[metric_key].dropna().iloc[-1]
                        if not data[metric_key].dropna().empty
                        else 0
                    )
                    all_metrics[metric_key].append(value)

            # Calculate means and standard errors
            for metric_key in metrics_map.values():
                metric_values = all_metrics[metric_key]
                values.append(np.mean(metric_values) * 100)
                errors.append((np.std(metric_values, ddof=1) / np.sqrt(len(metric_values))) * 100)

        plt.bar(
            x + i * width - width,
            values,
            width,
            yerr=errors,
            capsize=5,
            label=setting["label"],
            color=setting["color"],
        )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    plt.xticks(x, metrics_map.keys(), fontsize=18)
    sns.despine()
    plt.tight_layout()

    return fig, ax


def create_experiment4_1_loss_plot(experiment1_data, main_font_size=22):
    """Create loss plot showing two-hop no-CoT losses for all Experiment 4.1 settings"""
    plt.rcParams.update({"font.size": main_font_size})
    fig, ax = plt.subplots(figsize=(9, 6))

    # Define metrics and styling
    loss_columns = ["eval/2hop_no_cot_loss", "eval/2hop_no_cot_shuffled_loss"]
    styles = ["-", ":"]  # Solid for normal, dotted for shuffled
    api = wandb.Api()

    # Plot each setting
    for setting in EXPERIMENT4_1_SETTINGS:
        if setting["runs"] is None:
            # Use experiment1 data baseline (LLaMA runs)
            runs = fetch_runs_for_model(
                "experiments/fully_synthetic/configs/no_cot_and_cot.yaml", "LLaMAForCausalLM"
            )
        else:
            # Fetch specified runs
            runs = [api.run(run_path) for run_path in setting["runs"]]

        # Plot both normal and shuffled losses
        for col, style in zip(loss_columns, styles):
            all_data = []
            for run in runs:
                history = run.scan_history()
                df = pd.DataFrame(history)
                if not df.empty and col in df.columns:
                    df = df[["train/epoch", col]].dropna()
                    all_data.append(df)

            if all_data:
                combined_data = pd.concat(all_data, axis=0)
                label = f"{setting['label']}" if style == "-" else f"{setting['label']} (random)"
                sns.lineplot(
                    data=combined_data,
                    x="train/epoch",
                    y=col,
                    label=label,
                    color=setting["color"],
                    linestyle=style,
                    alpha=0.55 if style == "-" else 1.0,
                    linewidth=5,
                    errorbar="se",
                    ax=ax,
                    legend=False,
                )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Loss")
    ax.set_xlim(0, 1)
    ax.set_ylim(5.25, 6.5)
    plt.xticks(fontsize=20)
    sns.despine()
    plt.tight_layout()

    return fig, ax


# %%
# Experiment 1 data
experiment1_data = fetch_experiment1_data()
llama_runs = fetch_runs_for_model(
    "experiments/fully_synthetic/configs/no_cot_and_cot.yaml", "LLaMAForCausalLM"
)
qwen_runs = fetch_runs_for_model(
    "experiments/fully_synthetic/configs/no_cot_and_cot.yaml", "Qwen2ForCausalLM"
)
# %%

# Experiment 1 plots

## ACCURACY plot
fig, ax = create_experiment1_accuracy_plot(experiment1_data)
fig.savefig(FIGURE_DIR / "experiment1_accuracy.pdf", dpi=300, bbox_inches="tight")

# LOSS plot
fig, _ = create_experiment1_combined_loss_plot(llama_runs, qwen_runs)
fig.savefig(FIGURE_DIR / "experiment1_combined_loss.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%

# Print LaTeX color definitions


def hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    return tuple(int(hex_code.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))


model_name_to_color_name = {
    "LLaMA-3-8B-Instruct": "exp2_llama",
    "Qwen2.5-7B-Instruct": "exp2_qwen",
    "GPT-4o-mini": "exp2_gpt4o_mini",
    "GPT-4o": "exp2_gpt4o",
}
for model, color_name in model_name_to_color_name.items():
    hex_color = MODEL_COLORS[model]
    rgb_color = hex_to_rgb(hex_color)
    print(f"\\definecolor{{{color_name}}}{{RGB}}{{{rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}}}")


# %%
fig, ax = create_experiment4_1_accuracy_plot(experiment1_data)
fig.savefig(FIGURE_DIR / "experiment4_1_accuracy.pdf", dpi=300, bbox_inches="tight")
plt.show()
# %%
fig, ax = create_experiment4_1_loss_plot(experiment1_data)
fig.savefig(FIGURE_DIR / "experiment4_1_loss.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Print LaTeX color definitions for Experiment 4.1
for setting in EXPERIMENT4_1_SETTINGS:
    rgb_color = hex_to_rgb(setting["color"])
    print(f"\\cblock{{{rgb_color[0]}}}{{{rgb_color[1]}}}{{{rgb_color[2]}}}")

# %%

INTERVENTION_4_2_SETTINGS = [
    {
        "label": "Baseline",
        "color": "#F2A93B",
        "runs": None,  # Will use experiment1 data
    },
    {
        "label": "Logit lens",
        "color": "#C0DB84",
        "runs": [
            "sita/latent_reasoning/mj920jnx",
            "sita/latent_reasoning/qrsnkc6a",
            "sita/latent_reasoning/3zac8pg5",
        ],
    },
    {
        "label": "Embed lens",
        "color": "#88C7C3",
        "runs": [
            "sita/latent_reasoning/z197iirz",
            "sita/latent_reasoning/b6x0m1uh",
            "sita/latent_reasoning/unj1jnh0",
        ],
    },
]


def create_experiment4_2_accuracy_plot(experiment1_data, main_font_size=22):
    """Create accuracy plot for Experiment 4.2 results"""
    plt.rcParams.update({"font.size": main_font_size})
    fig, ax = plt.subplots(figsize=(9, 6))

    metrics_map = {
        "1st hop": "acc_a",
        "2nd hop": "acc_b",
        "Two-hop\nwith CoT": "acc_2hop_cot",
        "Two-hop\nwithout CoT": "acc_2hop_0shot_strict",
    }

    x = np.arange(len(metrics_map))
    width = 0.25  # Width of bars
    api = wandb.Api()

    # Plot bars for each setting
    for i, setting in enumerate(INTERVENTION_4_2_SETTINGS):
        values = []
        errors = []

        if setting["runs"] is None:
            # Use experiment1 data for baseline
            metrics = experiment1_data["LLaMA-3-8B-Instruct"]
            for metric_key in metrics_map.values():
                metric_values = metrics[metric_key]
                values.append(np.mean(metric_values) * 100)
                errors.append((np.std(metric_values, ddof=1) / np.sqrt(len(metric_values))) * 100)
        else:
            # Fetch data from W&B runs
            all_metrics = {metric: [] for metric in metrics_map.values()}
            for run_path in setting["runs"]:
                run = api.run(run_path)
                history = run.scan_history()
                data = pd.DataFrame(history)

                for metric_key in metrics_map.values():
                    value = (
                        data[metric_key].dropna().iloc[-1]
                        if not data[metric_key].dropna().empty
                        else 0
                    )
                    all_metrics[metric_key].append(value)

            # Calculate means and standard errors
            for metric_key in metrics_map.values():
                metric_values = all_metrics[metric_key]
                values.append(np.mean(metric_values) * 100)
                errors.append((np.std(metric_values, ddof=1) / np.sqrt(len(metric_values))) * 100)

        plt.bar(
            x + i * width - width,
            values,
            width,
            yerr=errors,
            capsize=5,
            label=setting["label"],
            color=setting["color"],
        )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    plt.xticks(x, metrics_map.keys(), fontsize=18)
    sns.despine()
    plt.tight_layout()

    return fig, ax


def create_experiment4_2_main_loss_plot(experiment1_data, main_font_size=22):
    """Create loss plot showing two-hop no-CoT losses for all Experiment 4.2 settings"""
    plt.rcParams.update({"font.size": main_font_size})
    fig, ax = plt.subplots(figsize=(6, 6))

    # Define metrics and styling
    loss_columns = ["eval/2hop_no_cot/orig_loss", "eval/2hop_no_cot_shuffled/orig_loss"]
    styles = ["-", ":"]  # Solid for normal, dotted for shuffled
    api = wandb.Api()

    # Plot each setting
    for setting in INTERVENTION_4_2_SETTINGS:
        if setting["runs"] is None:
            # Use experiment1 data baseline (LLaMA runs)
            runs = fetch_runs_for_model(
                "experiments/fully_synthetic/configs/no_cot_and_cot.yaml", "LLaMAForCausalLM"
            )
        else:
            # Fetch specified runs
            runs = [api.run(run_path) for run_path in setting["runs"]]

        # Plot both normal and shuffled losses
        for col, style in zip(loss_columns, styles):
            all_data = []
            for run in runs:
                history = run.scan_history()
                df = pd.DataFrame(history)
                if not df.empty and col in df.columns:
                    df = df[["train/epoch", col]].dropna()
                    all_data.append(df)
                elif not df.empty and col.replace("/orig_loss", "_loss") in df.columns:
                    col = col.replace("/orig_loss", "_loss")
                    df = df[["train/epoch", col]].dropna()
                    all_data.append(df)

            if all_data:
                combined_data = pd.concat(all_data, axis=0)
                label = f"{setting['label']}" if style == "-" else f"{setting['label']} (random)"
                sns.lineplot(
                    data=combined_data,
                    x="train/epoch",
                    y=col,
                    label=label,
                    color=setting["color"],
                    linestyle=style,
                    alpha=0.55 if style == "-" else 1.0,
                    linewidth=5,
                    errorbar="se",
                    ax=ax,
                    legend=False,
                )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Loss")
    ax.set_xlim(0, 1)
    ax.set_ylim(5, 7)
    plt.xticks(fontsize=20)
    sns.despine()
    plt.tight_layout()

    return fig, ax


def create_experiment4_2_auxiliary_loss_plots(main_font_size=22):
    """Create separate plots for logit lens and embed lens auxiliary losses"""
    plt.rcParams.update({"font.size": main_font_size})
    api = wandb.Api()

    # Create logit lens plot
    fig_logit, ax_logit = plt.subplots(figsize=(6, 6))
    logit_runs = [api.run(run_path) for run_path in INTERVENTION_4_2_SETTINGS[1]["runs"]]
    all_data = []
    for run in logit_runs:
        history = run.scan_history()
        df = pd.DataFrame(history)
        if not df.empty and "eval/2hop_no_cot/aux_loss" in df.columns:
            df = df[["train/epoch", "eval/2hop_no_cot/aux_loss"]].dropna()
            all_data.append(df)

    if all_data:
        combined_data = pd.concat(all_data, axis=0)
        sns.lineplot(
            data=combined_data,
            x="train/epoch",
            y="eval/2hop_no_cot/aux_loss",
            color=INTERVENTION_4_2_SETTINGS[1]["color"],
            linewidth=5,
            errorbar="se",
            ax=ax_logit,
        )

    ax_logit.set_xlabel("Epoch")
    ax_logit.set_ylabel("Test $\\mathcal{L}_{aux}$")
    ax_logit.set_xlim(0, 1)
    plt.xticks(fontsize=20)
    sns.despine()
    plt.tight_layout()

    # Create embed lens plot
    fig_embed, ax_embed = plt.subplots(figsize=(6, 6))
    embed_runs = [api.run(run_path) for run_path in INTERVENTION_4_2_SETTINGS[2]["runs"]]
    all_data = []
    for run in embed_runs:
        history = run.scan_history()
        df = pd.DataFrame(history)
        if not df.empty and "eval/2hop_no_cot/aux_loss" in df.columns:
            df = df[["train/epoch", "eval/2hop_no_cot/aux_loss"]].dropna()
            all_data.append(df)

    if all_data:
        combined_data = pd.concat(all_data, axis=0)
        sns.lineplot(
            data=combined_data,
            x="train/epoch",
            y="eval/2hop_no_cot/aux_loss",
            color=INTERVENTION_4_2_SETTINGS[2]["color"],
            linewidth=5,
            errorbar="se",
            ax=ax_embed,
        )

    ax_embed.set_xlabel("Epoch")
    ax_embed.set_ylabel("Test $\\mathcal{L}_{aux}$")
    ax_embed.set_xlim(0, 1)
    plt.xticks(fontsize=20)
    sns.despine()
    plt.tight_layout()

    return (fig_logit, ax_logit), (fig_embed, ax_embed)


# %%
# Experiment 4.2 plots

# Create and save all plots
fig, ax = create_experiment4_2_accuracy_plot(experiment1_data)
fig.savefig(FIGURE_DIR / "experiment4_2_accuracy.pdf", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = create_experiment4_2_main_loss_plot(experiment1_data)
fig.savefig(FIGURE_DIR / "experiment4_2_main_loss.pdf", dpi=300, bbox_inches="tight")
plt.show()

(fig_logit, ax_logit), (fig_embed, ax_embed) = create_experiment4_2_auxiliary_loss_plots()
fig_logit.savefig(FIGURE_DIR / "experiment4_2_logit_lens_loss.pdf", dpi=300, bbox_inches="tight")
fig_embed.savefig(FIGURE_DIR / "experiment4_2_embed_lens_loss.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Print LaTeX color definitions for Experiment 4.2
for setting in INTERVENTION_4_2_SETTINGS:
    rgb_color = hex_to_rgb(setting["color"])
    print(f"\\cblock{{{rgb_color[0]}}}{{{rgb_color[1]}}}{{{rgb_color[2]}}}")

# %%


def create_distractor_accuracy_plot(conditions, ylim=(0, 100), figsize=(9, 6), width=0.35):
    """Create accuracy plot comparing different distractor conditions"""
    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(figsize=figsize)

    # Plot bars
    x = np.arange(len(conditions))

    for i, condition in enumerate(conditions):
        values = []
        for run in condition["runs"]:
            history = run.scan_history()
            data = pd.DataFrame(history)
            acc = data["acc_2hop_0shot_strict"].dropna().iloc[-1]
            values.append(acc * 100)

        mean = np.mean(values)
        err = np.std(values) / np.sqrt(len(values))

        plt.bar(
            i,
            mean,
            width,
            yerr=err,
            capsize=5,
            label=condition["name"],
            color=condition["color"],
        )

    ax.set_ylim(*ylim)
    ax.set_ylabel("Two-hop No-CoT Accuracy (%)")
    plt.xticks(x, [c["name"] for c in conditions], fontsize=18)
    sns.despine()
    plt.tight_layout()

    return fig, ax


def create_distractor_loss_plot(conditions, ylim=None):
    """Create loss plot comparing different distractor conditions"""
    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot loss curves
    loss_columns = ["eval/2hop_no_cot_loss", "eval/2hop_no_cot_shuffled_loss"]
    styles = ["-", ":"]  # Solid for normal, dotted for shuffled

    for condition in conditions:
        for col, style in zip(loss_columns, styles):
            all_data = []
            for run in condition["runs"]:
                history = run.scan_history()
                df = pd.DataFrame(history)
                if not df.empty and col in df.columns:
                    df = df[["train/epoch", col]].dropna()
                    all_data.append(df)

            if all_data:
                combined_data = pd.concat(all_data, axis=0)
                label = f"{condition['name']}" if style == "-" else None
                sns.lineplot(
                    data=combined_data,
                    x="train/epoch",
                    y=col,
                    label=label,
                    color=condition["color"],
                    linestyle=style,
                    alpha=0.55 if style == "-" else 1.0,
                    linewidth=5,
                    errorbar="se",
                    ax=ax,
                )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Loss")
    ax.set_xlim(0, 1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.xticks(fontsize=20)
    sns.despine()
    plt.tight_layout()

    return fig, ax


# %%
colors = sns.color_palette("Set2")
conditions_1 = [
    {
        "name": "Same doc",
        "runs": fetch_runs_for_model(
            "experiments/arxiv/both_hops_samedoc.yaml", "LLaMAForCausalLM"
        ),
        "color": "#BF635F",
    },
    {
        "name": "Same doc\n+ related",
        "runs": [
            wandb.Api().run(f"sita/latent_reasoning/{run_id}")
            for run_id in ["5smx1snj", "6avxqvgp", "i3252kbb"]
        ],
        "color": "#C0DB84",
    },
    {
        "name": "Same doc\n+ 3 other facts",
        "runs": [
            wandb.Api().run(f"sita/latent_reasoning/{run_id}")
            for run_id in ["wb34csai", "syp53ayi", "armyvrca"]
        ],
        "color": "#C8D1F4",
    },
    {
        "name": "Same doc\n+ 10 other facts",
        "runs": [
            wandb.Api().run(f"sita/latent_reasoning/{run_id}")
            for run_id in ["1242r8gx", "6v0uedmc", "mqonmmin"]
        ],
        "color": "#8095E5",
    },
    {
        "name": "Separate docs",
        "runs": fetch_runs_for_model(
            "experiments/fully_synthetic/configs/no_cot_and_cot.yaml", "LLaMAForCausalLM"
        ),
        "color": "#F2A93B",
    },
]

# Create and save plots
fig, ax = create_distractor_accuracy_plot(conditions_1, figsize=(12, 6), width=0.5)
fig.savefig(FIGURE_DIR / "distractor_accuracy_1.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%

conditions_2 = [
    {
        "name": "Same doc\n+ 10 other facts",
        "runs": [
            wandb.Api().run(f"sita/latent_reasoning/{run_id}")
            for run_id in ["1242r8gx", "6v0uedmc", "mqonmmin"]
        ],
        "color": "#8095E5",
    },
    {
        "name": "Separate docs",
        "runs": fetch_runs_for_model(
            "experiments/fully_synthetic/configs/no_cot_and_cot.yaml", "LLaMAForCausalLM"
        ),
        "color": "#F2A93B",
    },
]

# Create and save plots
fig, ax = create_distractor_accuracy_plot(conditions_2, ylim=(0, 2), figsize=(6, 6), width=0.75)
fig.savefig(FIGURE_DIR / "distractor_accuracy_2.pdf", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = create_distractor_loss_plot(conditions_2, ylim=(5, 6.5))
fig.savefig(FIGURE_DIR / "distractor_loss_2.pdf", dpi=300, bbox_inches="tight")
plt.show()


# %%

ABLATION_SETTINGS = [
    {
        "name": "Full mixture",
        "config": "experiments/fully_synthetic/configs/no_cot_and_cot.yaml",
        "color": "#F2A93B",
    },
    {
        "name": "No-CoT mixture",
        "config": "experiments/fully_synthetic/configs/nocot.yaml",
        "color": "#C0DB84",
    },
    {
        "name": "Atomic-only",
        "config": "experiments/fully_synthetic/configs/atomic.yaml",
        "color": "#8095E5",
    },
]


def create_ablation_accuracy_plot(main_font_size=22):
    """Create accuracy plot comparing different training data mixtures"""
    plt.rcParams.update({"font.size": main_font_size})
    fig, ax = plt.subplots(figsize=(9, 6))

    metrics_map = {
        "1st hop": "acc_a",
        "2nd hop": "acc_b",
        "Two-hop\nwith CoT": "acc_2hop_cot",
        "Two-hop\nwithout CoT": "acc_2hop_0shot_strict",
    }

    x = np.arange(len(metrics_map))
    width = 0.25  # Width of bars

    # Plot bars for each setting
    for i, setting in enumerate(ABLATION_SETTINGS):
        runs = fetch_runs_for_model(setting["config"], "LLaMAForCausalLM")
        assert len(runs) == 3, f"Expected 3 runs for {setting['name']}, got {len(runs)}"

        metrics = process_runs(runs)
        values = []
        errors = []

        for metric_key in metrics_map.values():
            metric_values = metrics[metric_key]
            values.append(np.mean(metric_values) * 100)
            errors.append((np.std(metric_values, ddof=1) / np.sqrt(len(metric_values))) * 100)

        plt.bar(
            x + i * width - width,
            values,
            width,
            yerr=errors,
            capsize=5,
            label=setting["name"],
            color=setting["color"],
        )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    plt.xticks(x, metrics_map.keys(), fontsize=18)
    sns.despine()
    plt.tight_layout()

    return fig, ax


def create_ablation_loss_plot(main_font_size=22):
    """Create loss plot showing two-hop no-CoT losses for all ablation settings"""
    plt.rcParams.update({"font.size": main_font_size})
    fig, ax = plt.subplots(figsize=(6, 6))

    # Define metrics and styling
    loss_columns = ["eval/2hop_no_cot_loss", "eval/2hop_no_cot_shuffled_loss"]
    styles = ["-", ":"]  # Solid for normal, dotted for shuffled

    # Plot each setting
    for setting in ABLATION_SETTINGS:
        runs = fetch_runs_for_model(setting["config"], "LLaMAForCausalLM")
        assert len(runs) == 3, f"Expected 3 runs for {setting['name']}, got {len(runs)}"

        # Plot both normal and shuffled losses
        for col, style in zip(loss_columns, styles):
            all_data = []
            for run in runs:
                history = run.scan_history()
                df = pd.DataFrame(history)
                if not df.empty and col in df.columns:
                    df = df[["train/epoch", col]].dropna()
                    all_data.append(df)

            if all_data:
                combined_data = pd.concat(all_data, axis=0)
                label = f"{setting['name']}" if style == "-" else f"{setting['name']} (random)"
                sns.lineplot(
                    data=combined_data,
                    x="train/epoch",
                    y=col,
                    label=label,
                    color=setting["color"],
                    linestyle=style,
                    alpha=0.55 if style == "-" else 1.0,
                    linewidth=5,
                    errorbar="se",
                    ax=ax,
                    legend=False,
                )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Loss")
    ax.set_xlim(0, 1)
    # ax.set_ylim(5, 6.5)  # Same as experiment1 plots
    plt.xticks(fontsize=20)
    sns.despine()
    plt.tight_layout()

    return fig, ax


def create_ablation_cot_loss_plot(main_font_size=22):
    """Create loss plot showing two-hop CoT losses for all ablation settings"""
    plt.rcParams.update({"font.size": main_font_size})
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot each setting
    for setting in ABLATION_SETTINGS:
        runs = fetch_runs_for_model(setting["config"], "LLaMAForCausalLM")
        assert len(runs) == 3, f"Expected 3 runs for {setting['name']}, got {len(runs)}"

        all_data = []
        for run in runs:
            history = run.scan_history()
            df = pd.DataFrame(history)
            if not df.empty and "eval/2hop_cot_loss" in df.columns:
                df = df[["train/epoch", "eval/2hop_cot_loss"]].dropna()
                all_data.append(df)

        if all_data:
            combined_data = pd.concat(all_data, axis=0)
            sns.lineplot(
                data=combined_data,
                x="train/epoch",
                y="eval/2hop_cot_loss",
                label=setting["name"],
                color=setting["color"],
                linewidth=5,
                errorbar="se",
                ax=ax,
                legend=False,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Loss")
    ax.set_xlim(0, 1)
    plt.xticks(fontsize=20)
    sns.despine()
    plt.tight_layout()

    return fig, ax


# Add at the bottom with other plot generation code:
# %%
# Ablation study plots
fig, ax = create_ablation_accuracy_plot()
fig.savefig(FIGURE_DIR / "ablation_data_mixture_accuracy.pdf", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = create_ablation_loss_plot()
fig.savefig(FIGURE_DIR / "ablation_data_mixture_nocot_loss.pdf", dpi=300, bbox_inches="tight")
plt.show()
# %%
fig, ax = create_ablation_cot_loss_plot()
fig.savefig(FIGURE_DIR / "ablation_data_mixture_cot_loss.pdf", dpi=300, bbox_inches="tight")
plt.show()
# %%
# Print Latex color definitions of each setting
for setting in ABLATION_SETTINGS:
    hex_color = setting["color"]
    rgb_color = hex_to_rgb(hex_color)
    print(
        f"\\definecolor{{{setting['name']}}}{{RGB}}{{{rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}}}"
    )

# %%
