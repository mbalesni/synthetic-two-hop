# %%
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

MODEL_NAME_MAP = {
    "claude-3-haiku-20240307": "Claude 3 Haiku",
    "claude-3-opus-20240229": "Claude 3 Opus",
    "claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-opus-4-20250514": "Claude Opus 4",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "Llama 3.1 8B Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "Llama 3.1 70B Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "Llama 3.1 405B Instruct",
    "gpt-3.5-turbo-0125": "GPT-3.5-turbo",
    "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
    "gpt-4o-2024-05-13": "GPT-4o",
    "gpt-4.1-nano-2025-04-14": "GPT-4.1-nano",
    "gpt-4.1-mini-2025-04-14": "GPT-4.1-mini",
    "gpt-4.1-2025-04-14": "GPT-4.1",
    "gpt-4.5-preview-2025-02-27": "GPT-4.5",
    "Qwen/Qwen2.5-7B-Instruct-Turbo": "Qwen2.5-7B",
    "Qwen/Qwen2.5-72B-Instruct-Turbo": "Qwen2.5-72B",
}

def preprocess():

    df1 = pd.read_csv("two_hop_results_raw_hopping_too_late.csv")
    df2 = pd.read_csv("two_hop_results_raw_hopping_too_late_2.csv")
    df3 = pd.read_csv("two_hop_results_raw_hopping_too_late_new_ants.csv")
    df = pd.concat([df1, df2, df3])

    # Select the columns to keep from the original DataFrame
    columns_to_keep = [
        "model",
        "id",
        "e1_label",
        "e2_label",
        "e3_label",
        "e1_type",
        "e2_type",
        "e3_type",
        "r1_template",
        "r2_template",
        "source_prompt",
    ]

    # Create a DataFrame with the columns to keep
    df_kept_columns = df[columns_to_keep].drop_duplicates()

    # Perform the pivot operation
    df_pivot = df.pivot_table(
        values="correct",
        index=["model", "id"],
        columns="task",
    )

    # Merge the pivoted DataFrame with the kept columns
    df_merged = df_kept_columns.merge(df_pivot, on=["model", "id"])
    df_merged["both_1hops_correct"] = df_merged["one_hop_a"].astype(int) & df_merged[
        "one_hop_b"
    ].astype(int)
    df_merged["two_hop_also_correct"] = (
        df_merged["two_hop_no_cot"].astype(int) & df_merged["both_1hops_correct"]
    )
    df_merged["two_hop_also_correct_corrected"] = (
        df_merged["two_hop_no_cot"].astype(int)
        & df_merged["both_1hops_correct"]
        & ~df_merged["two_hop_no_cot_baseline1"].astype(int)
        & ~df_merged["two_hop_no_cot_baseline2"].astype(int)
    )

    print(len(df_merged))
    print(df_merged.columns)

    df_merged.to_csv("two_hop_results_raw_hopping_too_late_pivot.csv", index=False)

def hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    return tuple(int(hex_code.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))


def create_api_model_performance_plot():
    df1 = pd.read_csv("two_hop_results_hopping_too_late.csv")
    df2 = pd.read_csv("two_hop_results_hopping_too_late_2.csv")
    df3 = pd.read_csv("two_hop_results_hopping_too_late_new_ants.csv")
    df = pd.concat([df1, df2, df3])
    print(f"Length of df: {len(df)}")
    # Keep only triplets where (1) both one-hops are correct and (2) the model does not use reasoning shortcuts
    df = df[df["both_1hops_correct"] == 1.0]
    df = df[df["two_hop_no_cot_baseline1"] == 0]
    df = df[df["two_hop_no_cot_baseline2"] == 0]
    # df = df[df["e2_type"] == "country"]

    # Define the desired order of models and their human-friendly names

    model_order = list(MODEL_NAME_MAP.keys())
    model_names = list(MODEL_NAME_MAP.values())

    # Create a dictionary to map model IDs to human-friendly names

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
        "two_hop_also_correct_corrected",
        "two_hop_also_correct_corrected_sem",
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
        results["two_hop_also_correct_corrected"],
        width,
        yerr=results["two_hop_also_correct_corrected_sem"],
        capsize=5,
        label="No-CoT",
        color=colors[1],
    )

    # ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x_comp_gap)
    ax.set_xticklabels(
        [MODEL_NAME_MAP[model] for model in results.index], fontsize=14, rotation=45, ha="right"
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

preprocess()
fig, ax = create_api_model_performance_plot()
fig.savefig("experiment3_accuracy.pdf", dpi=300)
plt.show()

fig, ax = create_api_model_frequency_plot()
fig.savefig("experiment3_frequency.pdf", dpi=300)
plt.show()

# %%

for model in MODEL_NAME_MAP.keys():
    fig, ax = create_api_model_frequency_plot(model, model_in_title=True, xlim=110)
    model_without_slash = model.split("/")[-1]
    fig.savefig(f"experiment3_frequency_{model_without_slash}.pdf", dpi=300)
    plt.show()


# %%
df1 = pd.read_csv("two_hop_results_hopping_too_late.csv")
df2 = pd.read_csv("two_hop_results_hopping_too_late_2.csv")
df = pd.concat([df1, df2])
df.groupby("model").mean()[['both_1hops_correct', 'two_hop_cot', "two_hop_also_correct", "two_hop_also_correct_corrected"]]
# %%
