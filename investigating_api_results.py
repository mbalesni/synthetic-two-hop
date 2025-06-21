# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_accuracy_bar_chart(performance_data, title):
    """
    Creates a bar chart showing the accuracy for each question type category.

    Parameters:
    performance_data (pd.Series): A series containing accuracy values for different question types.
    title (str): The title of the bar chart.
    """
    # Sort the data by accuracy
    sorted_data = performance_data.sort_values(ascending=False)

    # Create the bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(sorted_data)), sorted_data.values, align="center", alpha=0.7)
    plt.xlabel("Question Type Category")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xticks(range(len(sorted_data))[::10], rotation=45, ha="right")
    plt.tight_layout()

    # Add value labels on top of each bar
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2., height,
    #              f'{height:.2f}',
    #              ha='center', va='bottom')

    plt.show()


# %%


def get_performance_for_model(model_name: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Process raw results data for a specific model and return performance metrics.

    Args:
        model_name: Name of the model to analyze
        df: Optional DataFrame to use. If None, loads from CSV

    Returns:
        DataFrame with performance metrics grouped by template types
    """
    if df is None:
        df = pd.read_csv("two_hop_results_raw_hopping_too_late.csv")
        print("Models: ", df["model"].unique())

    df = df[df["model"] == model_name]

    df = df.pivot_table(
        values="correct",
        index=["model", "id", "r1_template", "r2_template", "source_prompt", "e3_label"],
        columns="task",
    )
    df["both_1hops_correct"] = df["one_hop_a"].astype(int) & df["one_hop_b"].astype(int)
    df["two_hop_also_correct"] = df["two_hop_no_cot"].astype(int) & df["both_1hops_correct"]
    df["two_hop_also_correct_corrected"] = (
        df["two_hop_no_cot"].astype(int)
        & df["both_1hops_correct"]
        & ~df["two_hop_no_cot_baseline1"].astype(int)
        & ~df["two_hop_no_cot_baseline2"].astype(int)
    )

    performance = df.groupby(["r1_template", "r2_template"]).mean()
    performance = performance[performance["both_1hops_correct"] > 0.0]

    return performance


def plot_api_model_frequency(model_name: str):
    performance = get_performance_for_model(model_name)

    two_hop_also_correct_corrected = performance.groupby(["r1_template", "r2_template"])[
        "two_hop_also_correct_corrected"
    ].mean()
    two_hop_also_correct_corrected = two_hop_also_correct_corrected.sort_values(ascending=False)

    two_hop_cot = performance.groupby(["r1_template", "r2_template"])["two_hop_cot"].mean()
    two_hop_cot = two_hop_cot.sort_values(ascending=False)

    # Create the line chart
    plt.rcParams.update({"font.size": 24})
    plt.figure(figsize=(10, 7))

    # Use tab20c color map
    colors = plt.cm.tab20c([0, 2])  # Select two distinct colors from tab20c

    plt.plot(
        range(len(two_hop_cot)),
        two_hop_cot.values,
        marker="o",
        label="CoT",
        markersize=15,
        linewidth=5,
        color=colors[1],
    )
    plt.plot(
        range(len(two_hop_also_correct_corrected)),
        two_hop_also_correct_corrected.values,
        marker="o",
        label="No-CoT",
        markersize=15,
        linewidth=5,
        color=colors[0],
    )
    plt.xlabel("# of Question Types")
    plt.ylabel("Average Accuracy")
    # every ten labels
    plt.xticks(list(range(0, len(two_hop_cot), 5)))
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.title(f"Model: {model_name}")
    plt.tight_layout()
    plt.legend()

    plt.savefig("figure_0_api_frequency.pdf", dpi=300)
    plt.show()


# Heatmap of r1 vs r2 no-CoT accuracy for a particular model


def plot_template_heatmap(model_name: str):
    """
    Creates a heatmap showing the no-CoT accuracy for different combinations of r1 and r2 templates.

    Args:
        model_name: Name of the model to analyze
    """
    performance = get_performance_for_model(model_name)

    # Get unique templates
    r1_templates = sorted(set(idx[0].strip("{}") for idx in performance.index))
    r2_templates = sorted(set(idx[1].strip("{}") for idx in performance.index))

    # Create matrix for heatmap
    heatmap_data = np.full((len(r2_templates), len(r1_templates)), np.nan)

    # Fill matrix with accuracies
    for (r1, r2), row in performance.iterrows():
        r1_idx = r1_templates.index(r1.strip("{}"))
        r2_idx = r2_templates.index(r2.strip("{}"))
        heatmap_data[r2_idx, r1_idx] = row["two_hop_also_correct_corrected"]

    # Create heatmap with adjusted parameters
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        heatmap_data,
        xticklabels=r1_templates,
        yticklabels=r2_templates,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "No-CoT Accuracy"},
        annot_kws={"size": 6},  # Smaller annotation font
    )

    plt.title(f"Template Combination Accuracy for {model_name}", fontsize=10)
    plt.xlabel("R1 Template", fontsize=8)
    plt.ylabel("R2 Template", fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=6)  # Smaller x-axis labels
    plt.yticks(rotation=0, fontsize=6)  # Smaller y-axis labels
    plt.tight_layout()
    plt.show()


plot_api_model_frequency("claude-3-opus-20240229")

# %%
models = [
    "openai/gpt-3.5-turbo-0125",
    "openai/gpt-4o-2024-05-13",
    "openai/gpt-4o-mini-2024-07-18",
    "anthropic/claude-3-haiku-20240307",
    "anthropic/claude-3-sonnet-20240229",
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-opus-20240229",
    "together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
]
for model in models:
    model = model.split("/", 1)[1]
    print(f"{model}")
    plot_api_model_frequency(model)

# %%

# Get performance data for each model
df_4o = get_performance_for_model("gpt-4o-2024-05-13")
df_opus = get_performance_for_model("claude-3-opus-20240229")
df_405 = get_performance_for_model("meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo")

# Get the question types where two_hop_also_correct_corrected = 0 for each model
zero_perf_4o = df_4o[df_4o["two_hop_also_correct_corrected"] == 0].index
zero_perf_opus = df_opus[df_opus["two_hop_also_correct_corrected"] == 0].index
zero_perf_405 = df_405[df_405["two_hop_also_correct_corrected"] == 0].index

# Find common question types where all models fail
common_zeros = set(zero_perf_4o) & set(zero_perf_opus) & set(zero_perf_405)

# Get all question types (from any of the models)
all_types = set(df_4o.index) | set(df_opus.index) | set(df_405.index)
complement_types = all_types - common_zeros


def get_cot_acc(df, r1r2_tuple):
    if r1r2_tuple in df.index:
        return df.loc[r1r2_tuple, "two_hop_cot"]
    else:
        return float("nan")


print(f"\nFound {len(common_zeros)} question types where all models have zero performance:")
for qtype in sorted(common_zeros):
    r1, r2 = qtype[0].strip("{}"), qtype[1].strip("{}")
    cot_4o = get_cot_acc(df_4o, qtype)
    cot_opus = get_cot_acc(df_opus, qtype)
    cot_405 = get_cot_acc(df_405, qtype)
    print(f"{r2} -> {r1} [4o CoT: {cot_4o:.2f}, Opus CoT: {cot_opus:.2f}, 405 CoT: {cot_405:.2f}]")

print(f"\nFound {len(complement_types)} question types where at least one model succeeds:")
for qtype in sorted(complement_types):
    r1, r2 = qtype[0].strip("{}"), qtype[1].strip("{}")
    cot_4o = get_cot_acc(df_4o, qtype)
    cot_opus = get_cot_acc(df_opus, qtype)
    cot_405 = get_cot_acc(df_405, qtype)
    print(f"{r2} -> {r1} [4o CoT: {cot_4o:.2f}, Opus CoT: {cot_opus:.2f}, 405 CoT: {cot_405:.2f}]")


# Create a DataFrame for the heatmap
def create_cot_heatmap(relations, df_4o, df_opus, df_405):
    data = []
    for qtype in relations:
        r1, r2 = qtype[0].strip("{}"), qtype[1].strip("{}")
        relation = f"{r2} -> {r1}"
        cot_4o = get_cot_acc(df_4o, qtype)
        cot_opus = get_cot_acc(df_opus, qtype)
        cot_405 = get_cot_acc(df_405, qtype)
        data.append([relation, cot_4o, cot_opus, cot_405])

    print(data)
    df_heatmap = pd.DataFrame(data, columns=["Relation", "4o", "Opus", "405"])
    df_heatmap = df_heatmap.set_index("Relation")

    # Sort by max CoT accuracy
    df_heatmap["max_cot"] = df_heatmap.max(axis=1)
    df_heatmap = df_heatmap.sort_values("max_cot", ascending=False)
    df_heatmap = df_heatmap.drop("max_cot", axis=1)

    # Create heatmap
    plt.figure(figsize=(16, len(relations) * 0.8))
    sns.heatmap(
        df_heatmap,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "CoT Accuracy"},
    )
    plt.title("CoT Accuracy by Relation Type and Model")
    plt.tight_layout()
    plt.show()

    return df_heatmap


# Create separate heatmaps for zero performance and complement types
print("\nZero Performance Relations:")
zero_perf_df = create_cot_heatmap(common_zeros, df_4o, df_opus, df_405)

print("\nNon-zero Performance Relations:")
complement_df = create_cot_heatmap(complement_types, df_4o, df_opus, df_405)


# %%

# scatter plot
# 2hop cot
# 2hop no cot corrected


def plot_accuracy_distribution(model_name: str):
    performance = get_performance_for_model(model_name)

    # Get the accuracies
    cot_accuracies = performance["two_hop_cot"]
    nocot_accuracies = performance["two_hop_also_correct_corrected"]

    # Create the plot
    plt.figure(figsize=(16, 10))

    # Define bins - we want enough resolution to see the concentration at 0
    bins = np.linspace(0, 1, 20)

    # Plot histograms with weights to show percentages
    weights_nocot = np.ones_like(nocot_accuracies) * 100 / len(nocot_accuracies)
    weights_cot = np.ones_like(cot_accuracies) * 100 / len(cot_accuracies)

    plt.hist(
        nocot_accuracies,
        bins=bins,
        alpha=0.7,
        label="No-CoT",
        color=plt.cm.tab20c(0),
        weights=weights_nocot,
    )

    # Plot CoT histogram with only top edge visible
    plt.hist(
        cot_accuracies,
        bins=bins,
        alpha=0.7,
        label="CoT",
        color=plt.cm.tab20c(4),
        weights=weights_cot,
        histtype="step",  # This makes it show only the edges
        linewidth=2,  # Make the line thicker
    )

    plt.xlabel("Accuracy")
    plt.ylabel("% of Question Types")
    plt.title(f"Distribution of Accuracies for {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


# plot_accuracy_distribution("gpt-4o-2024-05-13")
plot_accuracy_distribution("claude-3-opus-20240229")


# %%


# Scatterplot of 2hop cot vs 2hop no cot corrected
def plot_cot_vs_nocot_scatter():
    all_data = []
    for model in models:
        model = model.split("/", 1)[1]
        performance = get_performance_for_model(model)
        performance["model"] = model
        all_data.append(performance)

    df = pd.concat(all_data)

    # Create scatter plot with single regression line
    plt.figure(figsize=(12, 10))

    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y=x")

    # Single regression plot for all data
    sns.regplot(
        data=df,
        x="two_hop_also_correct_corrected",
        y="two_hop_cot",
        scatter=True,
        scatter_kws={"alpha": 0.6},
        line_kws={"alpha": 0.8},
    )

    plt.xlabel("No-CoT Accuracy")
    plt.ylabel("CoT Accuracy")
    plt.title("CoT vs No-CoT Performance Across Models")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_cot_vs_nocot_scatter()

# %%


def compute_cot_vs_nocot_correlation():
    all_data = []
    for model in models:
        model = model.split("/", 1)[1]
        performance = get_performance_for_model(model)
        performance["model"] = model
        all_data.append(performance)

    df = pd.concat(all_data)

    # Compute correlations
    correlation = df["two_hop_also_correct_corrected"].corr(df["two_hop_cot"])
    spearman_corr = df["two_hop_also_correct_corrected"].corr(df["two_hop_cot"], method="spearman")

    print(f"Pearson correlation between CoT and No-CoT: {correlation:.3f}")
    print(f"Spearman correlation between CoT and No-CoT: {spearman_corr:.3f}")

    # Optional: Per-model correlations
    print("\nPer-model correlations:")
    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        pearson = model_data["two_hop_also_correct_corrected"].corr(model_data["two_hop_cot"])
        spearman = model_data["two_hop_also_correct_corrected"].corr(
            model_data["two_hop_cot"], method="spearman"
        )
        print(f"{model}:")
        print(f"  Pearson: {pearson:.3f}")
        print(f"  Spearman: {spearman:.3f}")


compute_cot_vs_nocot_correlation()

# %%


def compute_cot_vs_nocot_correlation():
    all_data = []
    for model in models:
        model = model.split("/", 1)[1]
        performance = get_performance_for_model(model)
        performance["model"] = model
        all_data.append(performance)

    df = pd.concat(all_data)

    # Regular correlations
    correlation = df["two_hop_also_correct_corrected"].corr(df["two_hop_cot"])
    spearman_corr = df["two_hop_also_correct_corrected"].corr(df["two_hop_cot"], method="spearman")

    # Binary correlations (0 vs non-zero)
    binary_nocot = (df["two_hop_also_correct_corrected"] > 0).astype(int)
    binary_cot = (df["two_hop_cot"] > 0).astype(int)
    binary_corr = binary_nocot.corr(binary_cot)
    binary_spearman = binary_nocot.corr(binary_cot, method="spearman")

    print("Overall correlations:")
    print(f"Pearson correlation between CoT and No-CoT: {correlation:.3f}")
    print(f"Spearman correlation between CoT and No-CoT: {spearman_corr:.3f}")
    print(f"Binary (0 vs non-zero) Pearson correlation: {binary_corr:.3f}")
    print(f"Binary (0 vs non-zero) Spearman correlation: {binary_spearman:.3f}")

    # Per-model correlations
    print("\nPer-model correlations:")
    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        pearson = model_data["two_hop_also_correct_corrected"].corr(model_data["two_hop_cot"])
        spearman = model_data["two_hop_also_correct_corrected"].corr(
            model_data["two_hop_cot"], method="spearman"
        )

        # Binary correlations per model
        binary_nocot_model = (model_data["two_hop_also_correct_corrected"] > 0).astype(int)
        binary_cot_model = (model_data["two_hop_cot"] > 0).astype(int)
        binary_corr_model = binary_nocot_model.corr(binary_cot_model)
        binary_spearman_model = binary_nocot_model.corr(binary_cot_model, method="spearman")

        print(f"{model}:")
        print(f"  Pearson: {pearson:.3f}")
        print(f"  Spearman: {spearman:.3f}")
        print(f"  Binary Pearson: {binary_corr_model:.3f}")
        print(f"  Binary Spearman: {binary_spearman_model:.3f}")


compute_cot_vs_nocot_correlation()


# %%
def compute_cot_vs_nocot_correlation():
    all_data = []
    for model in models:
        model = model.split("/", 1)[1]
        performance = get_performance_for_model(model)
        performance["model"] = model
        all_data.append(performance)

    df = pd.concat(all_data)

    # Regular correlations
    correlation = df["two_hop_also_correct_corrected"].corr(df["two_hop_cot"])
    spearman_corr = df["two_hop_also_correct_corrected"].corr(df["two_hop_cot"], method="spearman")

    # Binary no-CoT vs continuous CoT correlation
    binary_nocot = (df["two_hop_also_correct_corrected"] > 0).astype(int)
    point_biserial_corr = binary_nocot.corr(df["two_hop_cot"])

    print("Overall correlations:")
    print(f"Pearson correlation between CoT and No-CoT: {correlation:.3f}")
    print(f"Spearman correlation between CoT and No-CoT: {spearman_corr:.3f}")
    print(
        f"Point-biserial correlation (binary No-CoT vs continuous CoT): {point_biserial_corr:.3f}"
    )

    # Per-model correlations
    print("\nPer-model correlations:")
    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        pearson = model_data["two_hop_also_correct_corrected"].corr(model_data["two_hop_cot"])
        spearman = model_data["two_hop_also_correct_corrected"].corr(
            model_data["two_hop_cot"], method="spearman"
        )

        # Binary no-CoT vs continuous CoT correlation per model
        binary_nocot_model = (model_data["two_hop_also_correct_corrected"] > 0).astype(int)
        point_biserial_model = binary_nocot_model.corr(model_data["two_hop_cot"])

        print(f"{model}:")
        print(f"  Pearson: {pearson:.3f}")
        print(f"  Spearman: {spearman:.3f}")
        print(f"  Point-biserial (binary No-CoT vs continuous CoT): {point_biserial_model:.3f}")


compute_cot_vs_nocot_correlation()

# %%


def compute_nocot_performance_correlation():
    # Collect performance data for all models
    all_data = []
    for model in models:
        model_name = model.split("/", 1)[1]
        performance = get_performance_for_model(model_name)
        # Keep the index information
        performance = performance.reset_index()
        performance["model"] = model_name
        all_data.append(performance)

    # Concatenate all data
    df = pd.concat(all_data)

    # Choose a specific model for analysis
    model_of_interest = "gpt-4o-2024-05-13"
    df_model = df[df["model"] == model_of_interest]

    # Split based on r1_template to create two groups of question types
    unique_r1 = sorted(df_model["r1_template"].unique())
    split_point = len(unique_r1) // 2
    r1_subset_a = set(unique_r1[:split_point])

    # Create subsets based on r1_template
    df_model["subset"] = df_model["r1_template"].apply(lambda x: "A" if x in r1_subset_a else "B")

    # Calculate mean performance for each r2_template in each subset
    subset_performance = (
        df_model.groupby(["r2_template", "subset"])["two_hop_also_correct_corrected"]
        .mean()
        .unstack()
    )

    # Remove any r2_templates that don't have data in both subsets
    subset_performance = subset_performance.dropna()

    # Compute correlations between subsets
    pearson_corr = subset_performance["A"].corr(subset_performance["B"])
    spearman_corr = subset_performance["A"].corr(subset_performance["B"], method="spearman")

    print(f"Number of question types in analysis: {len(subset_performance)}")
    print(f"Correlation of No-CoT performance between subsets for {model_of_interest}:")
    print(f"  Pearson correlation: {pearson_corr:.3f}")
    print(f"  Spearman correlation: {spearman_corr:.3f}")

    # Visualization
    plt.figure(figsize=(10, 8))
    sns.regplot(
        x=subset_performance["A"],
        y=subset_performance["B"],
        scatter_kws={"alpha": 0.6},
        line_kws={"color": "red"},
    )
    plt.xlabel("No-CoT Accuracy on Subset A")
    plt.ylabel("No-CoT Accuracy on Subset B")
    plt.title(f"No-CoT Performance Correlation between Subsets\nfor {model_of_interest}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


compute_nocot_performance_correlation()
