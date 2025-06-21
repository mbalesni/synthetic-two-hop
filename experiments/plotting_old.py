# %%
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import wandb.apis.public.runs


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


def create_figure1_accuracy_plot(runs_by_setting, main_font_size, xtick_font_size, colors):
    plt.rcParams.update({"font.size": main_font_size})
    fig, ax = plt.subplots(figsize=(9, 6))

    accuracies = ["acc_a", "acc_b", "acc_2hop_cot", "acc_2hop_0shot_strict"]
    labels = [
        "1st hop",
        "2nd hop",
        "Two-hop\nwith CoT",
        "Two-hop\nwithout CoT",
    ]

    # Calculate means and standard errors across runs
    values_by_metric = {acc: [] for acc in accuracies}
    for run in runs_by_setting:
        history = run.scan_history()
        data = pd.DataFrame(history)
        for acc in accuracies:
            value = data[acc].dropna().iloc[-1] if not data[acc].dropna().empty else 0
            values_by_metric[acc].append(value * 100)  # Convert to percentage

    means = [np.mean(values_by_metric[acc]) for acc in accuracies]
    errors = [np.std(values_by_metric[acc]) / np.sqrt(len(runs_by_setting)) for acc in accuracies]

    ax.bar(labels, means, yerr=errors, capsize=5, color=colors)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    plt.xticks(fontsize=xtick_font_size)
    sns.despine()
    plt.tight_layout()

    return fig, ax


def create_figure1_loss_plot(runs_by_setting, main_font_size, xtick_font_size, colors, labels):
    plt.rcParams.update({"font.size": main_font_size})
    fig, ax = plt.subplots(figsize=(9, 6))

    loss_columns = [
        "eval/2hop_no_cot_loss",
        "eval/2hop_no_cot_shuffled_loss",
    ]
    styles = ["-", ":"]
    colors = colors[3:]

    # For each metric, collect data from all runs
    for col, label, color, style in zip(loss_columns, labels, colors, styles):
        all_data = []
        for run in runs_by_setting:
            history = run.scan_history()
            df = pd.DataFrame(history)
            if not df.empty and col in df.columns:
                df = df[["train/epoch", col]].dropna()
                all_data.append(df)

        if all_data:
            # Combine data from all runs
            combined_data = pd.concat(all_data, axis=0)
            # Use seaborn's lineplot which automatically calculates and shows confidence intervals
            sns.lineplot(
                data=combined_data,
                x="train/epoch",
                y=col,
                label=label,
                color=color,
                linestyle=style,
                linewidth=5,
                errorbar="se",  # Show standard error
                ax=ax,
                legend=False,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test No-CoT Loss")
    ax.set_xlim(0, 1)
    ax.set_ylim(5.25, 6.5)
    plt.xticks(fontsize=xtick_font_size)
    sns.despine()
    plt.tight_layout()

    return fig, ax


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


# Set up wandb API
api = wandb.Api()

# %%
# Experiment 1: Llama only

# Fetch runs for "Two hop no-cot and cot" setting
runs = fetch_runs_for_setting(
    project_name="sita/latent_reasoning",
    config_path="experiments/fully_synthetic/configs/no_cot_and_cot.yaml",
)
# filter runs to only LLaMA-3-8B-Instruct
runs = [
    run
    for run in runs
    if isinstance(run.config.get("architectures"), list)
    and any("llama" in arch.lower() for arch in run.config["architectures"])
]

# Create and save plots
# colors = list(reversed((sns.color_palette("tab20c", 4)[:4]))) + ["red"]
colors = ["#FCEACF", "#F9D59F", "#F2A93B", "#F46920", "#000000"]
labels = [
    "First one-hop",
    "Second one-hop",
    "Two-hop with CoT",
    "Two-hop without CoT",
    "Two-hop without CoT (shuffled)",  # TODO: rename "shuffled to "random"
]
loss_fig, _ = create_figure1_loss_plot(
    runs, main_font_size=22, xtick_font_size=20, colors=colors, labels=labels
)
loss_fig.savefig("figure1_loss.pdf", dpi=300)
plt.show()

print("colors for fig1")
rgb_colors = [
    (int(r * 255), int(g * 255), int(b * 255))
    for r, g, b in list(reversed((sns.color_palette("tab20c", 4)[:4])))
]
print(rgb_colors)

accuracy_fig, _ = create_figure1_accuracy_plot(
    runs, main_font_size=22, xtick_font_size=18, colors=colors
)
accuracy_fig.savefig("figure1_accuracy.pdf", dpi=300)
plt.show()
# %%
# Experiment 2: Same document & In-context

fig, ax = create_figure1_three_settings_plot()
fig.savefig("figure1_three_settings.pdf", dpi=300)
plt.show()

# %%
# Experiment 3: API model results


def hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    return tuple(int(hex_code.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))


def create_api_model_performance_plot():
    df = pd.read_csv("two_hop_results_raw_hopping_too_late_pivot.csv")
    # Keep only triplets where (1) both one-hops are correct and (2) the model does not use reasoning shortcuts
    df = df[df["both_1hops_correct"] == 1.0]
    df = df[df["two_hop_no_cot_baseline1"] == 0]
    df = df[df["two_hop_no_cot_baseline2"] == 0]
    # df = df[df["e2_type"] == "country"]

    # Define the desired order of models and their human-friendly names
    model_order = [
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "gpt-3.5-turbo-0125",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-05-13",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    ]
    model_names = [
        "Claude 3 Haiku",
        "Claude 3 Sonnet",
        "Claude 3 Opus",
        "GPT-3.5-turbo",
        "GPT-4o-mini",
        "GPT-4o",
        "Llama 3.1 8B Instruct",
        "Llama 3.1 70B Instruct",
        "Llama 3.1 405B Instruct",
    ]

    # Create a dictionary to map model IDs to human-friendly names
    model_name_map = dict(zip(model_order, model_names))

    # Filter and sort the dataframe
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values("model")

    # Group by model and calculate mean and standard error
    # First, remove non-numeric columns:
    df_pre_group = df.copy()
    numeric_cols = df_pre_group.select_dtypes(include=[np.number]).columns
    cols_to_keep = list(numeric_cols) + ["model"]
    df_pre_group = df_pre_group[cols_to_keep]
    results: pd.DataFrame = df_pre_group.groupby("model").mean()
    results["fraction"] = results["two_hop_also_correct"] / results["two_hop_cot"]
    sem: pd.DataFrame = df_pre_group.groupby("model").sem()
    results = results.join(sem, rsuffix="_sem")

    # Convert accuracies to percentages
    for col in [
        "two_hop_cot",
        "two_hop_also_correct",
        "two_hop_cot_sem",
        "two_hop_also_correct_sem",
    ]:
        results[col] *= 100

    x_comp_gap = np.arange(len(results))
    width = 0.35

    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = [
        "#F2A93B",
        "#F46920",
    ]
    # Print Latex color definition for CoT / No-CoT:
    for label, color in zip(["exp3_with_cot", "exp3_without_cot"], colors):
        rgb_color = hex_to_rgb(color)
        print(f"\\definecolor{{{label}}}{{RGB}}{{{rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}}}")

    ax.bar(
        x_comp_gap - width / 2,
        results["two_hop_cot"],
        width,
        yerr=results["two_hop_cot_sem"],
        capsize=5,
        label="CoT",
        color=colors[0],
    )
    ax.bar(
        x_comp_gap + width / 2,
        results["two_hop_also_correct"],
        width,
        yerr=results["two_hop_also_correct_sem"],
        capsize=5,
        label="No-CoT",
        color=colors[1],
    )

    # ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x_comp_gap)
    ax.set_xticklabels(
        [model_name_map[model] for model in results.index], fontsize=14, rotation=45, ha="right"
    )
    # remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    return fig, ax


def create_api_model_frequency_plot(
    model_name: str = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    model_in_title: bool = False,
    xlim: int | None = None,
):
    df = pd.read_csv("two_hop_results_raw_hopping_too_late_pivot.csv")
    print("Models: ", df["model"].unique())

    df = df[df["model"] == model_name]
    # Keep only triplets where (1) both one-hops are correct and (2) the model does not use reasoning shortcuts
    df = df[df["both_1hops_correct"] == 1]
    df = df[df["two_hop_no_cot_baseline1"] == 0]
    df = df[df["two_hop_no_cot_baseline2"] == 0]

    group_key = ["r1_template", "r2_template"]
    # remove non-numeric columns, but KEEP non-numeric ones we want to group by:
    df_pre_group = df.copy()
    # Keep only numeric columns except those in group_key
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_keep = list(numeric_cols) + group_key
    df_pre_group = df_pre_group[cols_to_keep]
    performance = df_pre_group.groupby(group_key).mean()

    # no-Cot
    two_hop_no_cot = performance.groupby(group_key)["two_hop_also_correct"].mean()
    two_hop_no_cot = two_hop_no_cot.sort_values(ascending=False)

    two_hop_cot = performance.groupby(group_key)["two_hop_cot"].mean()
    two_hop_cot = two_hop_cot.sort_values(ascending=False)

    # Create the line chart
    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = [
        "#F2A93B",
        "#F46920",
    ]

    ax.plot(
        range(len(two_hop_cot)),
        two_hop_cot.values * 100,  # Convert to percentage
        marker="o",
        label="CoT",
        markersize=15,
        linewidth=5,
        color=colors[1],
    )
    ax.plot(
        range(len(two_hop_no_cot)),
        two_hop_no_cot.values * 100,  # Convert to percentage
        marker="o",
        label="No-CoT",
        markersize=15,
        linewidth=5,
        color=colors[0],
    )

    ax.set_xlabel("Question Categories")
    ax.set_ylabel("Average Accuracy (%)")  # Updated label
    if model_in_title:
        model_without_slash = model_name.split("/")[-1]
        ax.set_title(f"{model_without_slash}")

    if xlim:
        ax.set_xlim(-5, xlim)
        ax.set_xticks(list(range(0, xlim, 10)))
    else:
        ax.set_xticks(list(range(0, len(two_hop_cot), 10)))

    # Remove grid and spines to match accuracy plot style
    ax.grid(False)
    sns.despine()
    plt.tight_layout()

    return fig, ax


fig, ax = create_api_model_performance_plot()
fig.savefig("experiment3_accuracy.pdf", dpi=300)
plt.show()

fig, ax = create_api_model_frequency_plot()
fig.savefig("experiment3_frequency.pdf", dpi=300)
plt.show()

# %%

for model in [
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
    "gpt-3.5-turbo-0125",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
]:
    fig, ax = create_api_model_frequency_plot(model, model_in_title=True, xlim=110)
    model_without_slash = model.split("/")[-1]
    fig.savefig(f"experiment3_frequency_{model_without_slash}.pdf", dpi=300)
    plt.show()


# %%
# WIP:
def create_nocot_accuracy_heatmap():
    # Select the frontier models
    frontier_models = [
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "claude-3-opus-20240229",
        "gpt-4o-2024-05-13",
    ]
    model_display_names = {
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "LLaMA-3-405B",
        "claude-3-opus-20240229": "Claude 3 Opus",
        "gpt-4o-2024-05-13": "GPT-4o",
    }

    df = pd.read_csv("two_hop_results_raw_hopping_too_late_pivot.csv")

    results = []

    for model in frontier_models:
        model_df = df[df["model"] == model]

        # all columns:
        with pd.option_context("display.max_columns", None):
            # make r1_template and r2_template columns:
            print(model_df.iloc[:3])

        # Keep only triplets where (1) both one-hops are correct and (2) the model does not use reasoning shortcuts
        model_df = model_df[model_df["both_1hops_correct"] == 1.0]
        model_df = model_df[model_df["two_hop_no_cot_baseline1"] == 0]
        model_df = model_df[model_df["two_hop_no_cot_baseline2"] == 0]

        # Keep only numeric columns except those in group_key
        group_key = ["r1_template", "r2_template"]
        numeric_cols = model_df.select_dtypes(include=[np.number]).columns
        cols_to_keep = list(numeric_cols) + group_key
        df_pre_group = model_df[cols_to_keep]
        per_category = df_pre_group.groupby(group_key).mean()

        # Before aggregating, get counts per category
        category_counts = model_df.groupby(["r1_template", "r2_template"]).size()

        # Store results for each category
        for idx, row in per_category.iterrows():
            r1 = idx[0]
            r2 = idx[1]
            count = category_counts[idx]  # Get count for this category
            category = (
                f"{r2.replace('{', '').replace('}', '')} → {r1.replace('{', '').replace('}', '')}"
            )
            category_str = f"{r2.replace('{', '').replace('}', '')} → {r1.replace('{', '').replace('}', '')} (n={count})"
            print(f"{model=}, {category=}, {row['two_hop_also_correct_corrected']=}")
            results.append(
                {
                    "category": category,
                    "category_str": category_str,
                    "model": model_display_names[model],
                    "accuracy": row["two_hop_also_correct_corrected"] * 100,
                }
            )

    print(results)
    # Convert to DataFrame and pivot to get desired format
    results_df = pd.DataFrame(results)
    table_df = results_df.pivot(index="category", columns="model", values="accuracy")

    # Add mean column and sort by it
    table_df["Mean"] = table_df.mean(axis=1)
    table_df = table_df.sort_values("Mean", ascending=False)

    # Drop rows where any model has NaN accuracy
    table_df = table_df.dropna(subset=table_df.columns.difference(["Mean"]), how="any")

    # Create heatmap
    plt.figure(figsize=(12, len(table_df) * 0.4))
    sns.heatmap(
        table_df,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",  # Red-Yellow-Green colormap
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Accuracy (%)"},
        mask=table_df.isna(),  # Mask NaN values to keep them visible
    )
    plt.title("No-CoT Accuracy by Category and Model")
    plt.tight_layout()
    plt.savefig("nocot_accuracy_heatmap.pdf", bbox_inches="tight")
    plt.show()

    return table_df


create_nocot_accuracy_heatmap()


# %%
# Find examples where all models fail without CoT but succeed with CoT
def find_interesting_examples():
    df = pd.read_csv("two_hop_results_raw_hopping_too_late_pivot.csv")

    # Filter for the three models we're interested in
    models_of_interest = [
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "claude-3-opus-20240229",
        "gpt-4o-2024-05-13",
    ]

    # Keep only rows where:
    # 1. both_1hops_correct is 1 (knows atomic facts)
    # 2. two_hop_cot is 1 (succeeds with CoT)
    # 3. two_hop_also_correct is 0 (fails without CoT)
    interesting_examples = []

    for model in models_of_interest:
        model_df = df[df["model"] == model]
        filtered = model_df[
            (model_df["both_1hops_correct"] == 1)
            & (model_df["two_hop_cot"] == 1)
            & (model_df["two_hop_also_correct"] == 0)
        ]
        if len(filtered) > 0:
            interesting_examples.append(set(filtered["id"]))

    # Find intersection of failures across all models
    common_failures = set.intersection(*interesting_examples)

    # Get the full questions for these examples
    example_rows = df[df["id"].isin(common_failures)].drop_duplicates("id")

    print(
        f"\nFound {len(common_failures)} examples where all models fail without CoT but succeed with CoT:\n"
    )
    for _, row in example_rows.head(10).iterrows():
        print(f"Question ID: {row['id']}")
        print(f"Question: {row['source_prompt']} --> {row['e3_label']}")
        print("---")


find_interesting_examples()
