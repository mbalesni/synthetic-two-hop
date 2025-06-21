# %%

import logging
import os
import re
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

import latent_reasoning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_triplets(model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo") -> pd.DataFrame:
    """Load triplets with performance data."""
    root_dir = os.path.dirname(latent_reasoning.__path__[0])
    df = pd.read_csv(f"{root_dir}/two_hop_results_raw_hopping_too_late_pivot.csv")
    df = df[df["model"] == model_name]

    # Filter for correct one-hops and no shortcuts
    df = df[df["both_1hops_correct"] == 1]
    df = df[df["two_hop_no_cot_baseline1"] == 0]
    df = df[df["two_hop_no_cot_baseline2"] == 0]
    df = df[df["two_hop_cot"] == 1]

    logger.info(f"Loaded {len(df)} valid triplets")
    return df


def entity_to_filename(entity: str) -> str:
    """Convert arbitrary string to filename safely."""
    return re.sub(r"[^\w\-_]", "_", entity)


def compute_all_similarities(
    activations_path: str,
    label_a_col: str,
    label_b_col: str,
    triplets_df: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Compute similarities between all entity pairs and their performance."""
    logger.info("Loading activations...")
    a_activations = {
        entity: torch.load(f"{activations_path}/{entity_to_filename(entity)}.pt")
        if os.path.exists(f"{activations_path}/{entity_to_filename(entity)}.pt")
        else None
        for entity in tqdm(triplets_df[label_a_col].unique())
    }
    logger.info(f"Loaded {len(a_activations)} activations for {label_a_col}")
    b_activations = {
        entity: torch.load(f"{activations_path}/{entity_to_filename(entity)}.pt")
        if os.path.exists(f"{activations_path}/{entity_to_filename(entity)}.pt")
        else None
        for entity in tqdm(triplets_df[label_b_col].unique())
    }
    if n_missing_activations := sum(1 for v in a_activations.values() if v is None):
        logger.warning(
            f"Warning: {n_missing_activations} activations for {label_a_col} are missing"
        )
    if n_missing_activations := sum(1 for v in b_activations.values() if v is None):
        logger.warning(
            f"Warning: {n_missing_activations} activations for {label_b_col} are missing"
        )
    # Filter out any entities that don't have activations
    a_activations = {k: v for k, v in a_activations.items() if v is not None}
    b_activations = {k: v for k, v in b_activations.items() if v is not None}
    logger.info(f"Loaded {len(b_activations)} activations for {label_b_col}")

    # Get all layer names and move tensors to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_names = list(next(iter(a_activations.values())).keys())

    # Pre-allocate arrays for all pairs
    all_similarities = []
    all_performances = []

    # Process one pair at a time
    for a, a_acts in tqdm(a_activations.items(), desc="Processing pairs"):
        relevant_rows = triplets_df[triplets_df[label_a_col] == a]

        # Stack all source layers for entity a
        a_stacked = torch.stack(
            [a_acts[layer].to(device) for layer in layer_names]
        )  # [n_layers, hidden_dim]

        for _, row in relevant_rows.iterrows():
            b = row[label_b_col]
            if b not in b_activations:
                continue

            # Stack all target layers for entity b
            b_stacked = torch.stack(
                [b_activations[b][layer].to(device) for layer in layer_names]
            )  # [n_layers, hidden_dim]

            # Compute cosine similarity for all layer combinations at once
            # Reshape a_stacked to [n_layers, 1, hidden_dim] and b_stacked to [1, n_layers, hidden_dim]
            cos_sims = torch.nn.functional.cosine_similarity(
                a_stacked.unsqueeze(1),  # [n_layers, 1, hidden_dim]
                b_stacked.unsqueeze(0),  # [1, n_layers, hidden_dim]
                dim=2,  # Compare along hidden dimension
            )  # Result: [n_layers, n_layers]

            # Compute euclidean distances for all layer combinations
            euc_dists = torch.cdist(
                a_stacked.unsqueeze(1),  # [n_layers, 1, hidden_dim]
                b_stacked.unsqueeze(0),  # [1, n_layers, hidden_dim]
            ).squeeze()  # [n_layers, n_layers]
            euc_proxs = 1 / (1 + euc_dists)

            # Store results
            all_similarities.append((cos_sims.cpu(), euc_proxs.cpu()))
            all_performances.append(row["two_hop_no_cot"])

    # Stack all similarities and performances
    all_cos_sims = torch.stack([cos for cos, _ in all_similarities])
    all_euc_proxs = torch.stack([euc for _, euc in all_similarities])
    all_performances = torch.tensor(all_performances)

    return all_cos_sims, all_euc_proxs, all_performances, layer_names


def compute_correlation_matrices(
    all_cos_sims: torch.Tensor,
    all_euc_proxs: torch.Tensor,
    all_performances: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute correlation matrices from pre-computed similarities."""
    n_layers = all_cos_sims.shape[1]
    cosine_matrix = torch.zeros((n_layers, n_layers))
    euclidean_matrix = torch.zeros((n_layers, n_layers))

    # Compute correlations for each layer combination
    for src_idx in range(n_layers):
        for tgt_idx in range(n_layers):
            if len(all_performances) > 0:
                cosine_matrix[src_idx, tgt_idx] = torch.corrcoef(
                    torch.stack([all_cos_sims[:, src_idx, tgt_idx], all_performances])
                )[0, 1]
                euclidean_matrix[src_idx, tgt_idx] = torch.corrcoef(
                    torch.stack([all_euc_proxs[:, src_idx, tgt_idx], all_performances])
                )[0, 1]

    return cosine_matrix, euclidean_matrix


def plot_correlation_matrices(
    cosine_matrix: np.ndarray,
    euclidean_matrix: np.ndarray,
    layer_names: list[str],
    source_label: str,
    target_label: str,
) -> tuple[plt.Figure, plt.Figure]:
    """Plot heatmaps for the correlation matrices."""
    fig_cos, ax_cos = plt.subplots(figsize=(36, 34))
    fig_euc, ax_euc = plt.subplots(figsize=(36, 34))

    cosine_max = cosine_matrix.max()
    euclidean_max = euclidean_matrix.max()

    # Plot cosine similarity correlations
    sns.heatmap(
        cosine_matrix,
        ax=ax_cos,
        cmap="RdYlBu",  # Red (negative) to Blue (positive)
        center=0,  # Center colormap at 0
        annot=True,
        fmt=".2f",
        xticklabels=layer_names,
        yticklabels=layer_names,
    )
    ax_cos.set_title(
        f"Correlation between Cosine Similarity and No-CoT Performance ({cosine_max:.2f})",
        fontsize=20,
    )
    ax_cos.set_xlabel(f"Target Layer ({target_label})")
    ax_cos.set_ylabel(f"Source Layer ({source_label})")

    # Plot euclidean proximity correlations
    sns.heatmap(
        euclidean_matrix,
        ax=ax_euc,
        cmap="RdYlBu",  # Now using same colormap as cosine (not reversed)
        center=0,
        annot=True,
        fmt=".2f",
        xticklabels=layer_names,
        yticklabels=layer_names,
    )
    ax_euc.set_title(
        f"Correlation between Euclidean Proximity and No-CoT Performance ({euclidean_max:.2f})",
        fontsize=20,
    )
    ax_euc.set_xlabel(f"Target Layer ({target_label})")
    ax_euc.set_ylabel(f"Source Layer ({source_label})")

    plt.tight_layout()
    return fig_cos, fig_euc


def plot_similarity_matrices(
    all_cos_sims: torch.Tensor,
    all_euc_proxs: torch.Tensor,
    layer_names: list[str],
    source_label: str,
    target_label: str,
) -> tuple[plt.Figure, plt.Figure]:
    """Plot heatmaps of average similarities across all pairs."""
    # Compute mean similarities across all pairs
    mean_cos_sims = all_cos_sims.mean(dim=0)  # [n_layers, n_layers]
    mean_euc_proxs = all_euc_proxs.mean(dim=0)  # [n_layers, n_layers]
    # Convert from BF16 to float32
    mean_cos_sims = mean_cos_sims.float()
    mean_euc_proxs = mean_euc_proxs.float()

    fig_cos, ax_cos = plt.subplots(figsize=(36, 34))
    fig_euc, ax_euc = plt.subplots(figsize=(36, 34))

    # Plot mean cosine similarities
    sns.heatmap(
        mean_cos_sims,
        ax=ax_cos,
        cmap="RdYlBu",
        center=0,  # Center at 0 since cosine similarity ranges from -1 to 1
        annot=True,
        fmt=".2f",
        xticklabels=layer_names,
        yticklabels=layer_names,
    )
    ax_cos.set_title("Average Cosine Similarity Between Layers")
    ax_cos.set_xlabel(f"Target Layer ({target_label})")
    ax_cos.set_ylabel(f"Source Layer ({source_label})")

    # Plot mean euclidean proximities
    sns.heatmap(
        mean_euc_proxs,
        ax=ax_euc,
        cmap="RdYlBu",
        vmin=0,  # Euclidean proximity is always positive
        vmax=mean_euc_proxs.max(),
        annot=True,
        fmt=".2f",
        xticklabels=layer_names,
        yticklabels=layer_names,
    )
    ax_euc.set_title("Average Euclidean Proximity Between Layers")
    ax_euc.set_xlabel(f"Target Layer ({target_label})")
    ax_euc.set_ylabel(f"Source Layer ({source_label})")

    plt.tight_layout()
    return fig_cos, fig_euc


# %%

# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"

current_dir = os.path.dirname(os.path.abspath(__file__))

# Analyze correlations
# for source_label, target_label in [("e1_label", "e3_label"), ("e2_label", "e3_label"), ("e1_label", "e2_label")]:
# for model_name in ["meta-llama/Meta-Llama-3.1-70B-Instruct"]:
for model_name in ["meta-llama/Meta-Llama-3.1-405B-Instruct"]:
    triplets_df = load_triplets(model_name + "-Turbo")
    model_name_pathsafe = model_name.replace("/", "_")
    print()
    print(f"Analyzing {model_name}")

    for source_label, target_label in [
        ("e1_label", "e1_label"),
        ("e1_label", "e2_label"),
        ("e1_label", "e3_label"),
        ("e2_label", "e2_label"),
        ("e2_label", "e3_label"),
        ("e3_label", "e3_label"),
    ]:
        all_cos_sims, all_euc_proxs, all_performances, layer_names = compute_all_similarities(
            f"{current_dir}/activations/{model_name_pathsafe}",
            source_label,
            target_label,
            triplets_df,
        )
        cosine_matrix, euclidean_matrix = compute_correlation_matrices(
            all_cos_sims, all_euc_proxs, all_performances
        )
        source_label_name = source_label.split("_")[0]
        target_label_name = target_label.split("_")[0]
        print(
            f"Correlation of Performance vs Similarity between {source_label_name} and {target_label_name}"
        )
        print(f"Cosine: {cosine_matrix.max()}, Euclidean: {euclidean_matrix.max()}")
        indices = np.unravel_index(cosine_matrix.argmax(), cosine_matrix.shape)
        print(
            f"[{source_label_name}] layer: {indices[0]} | [{target_label_name}] layer: {indices[1]} | Total layers: {len(layer_names)}"
        )
        # Plot "similarity vs performance" correlation matrices
        # fig_cos, fig_euc = plot_correlation_matrices(
        #     cosine_matrix, euclidean_matrix, layer_names, source_label, target_label
        # )
        plt.show()
        print()
# %%


# %%

# %%

print("Cosine matrix max:", cosine_matrix.max())
print("Euclidean matrix max:", euclidean_matrix.max())

# %%


def analyze_pair_similarities(
    activations_path: str,
    triplets_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze similarities for individual e2->e3 pairs.
    Returns a DataFrame with max similarities and performance for each pair.
    """
    # Load activations for e2 and e3
    e2_activations = {
        entity: torch.load(f"{activations_path}/{entity_to_filename(entity)}.pt")
        for entity in tqdm(triplets_df["e2_label"].unique())
    }
    e3_activations = {
        entity: torch.load(f"{activations_path}/{entity_to_filename(entity)}.pt")
        for entity in tqdm(triplets_df["e3_label"].unique())
    }

    # Get all layer names
    layer_names = list(next(iter(e2_activations.values())).keys())

    # Store results for each pair
    pair_results = []

    # Group by e2->e3 pairs and compute average performance
    pair_performance = (
        triplets_df.groupby(["e2_label", "e3_label"])["two_hop_no_cot"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # For each e2->e3 pair
    for _, row in tqdm(pair_performance.iterrows(), total=len(pair_performance), desc="Pairs"):
        e2, e3 = row["e2_label"], row["e3_label"]

        # Skip if we don't have activations for either entity
        if e2 not in e2_activations or e3 not in e3_activations:
            print(f"Skipping {e2} -> {e3} because we don't have activations")
            continue

        # Initialize max similarities
        max_cos_sim = float("-inf")
        max_euc_proximity = float("-inf")

        # Check all layer combinations
        for src_layer in layer_names:
            for tgt_layer in layer_names:
                e2_vec = e2_activations[e2][src_layer]
                e3_vec = e3_activations[e3][tgt_layer]

                # Compute similarities
                cos_sim = torch.nn.functional.cosine_similarity(
                    e2_vec.unsqueeze(0), e3_vec.unsqueeze(0)
                ).item()
                euc_dist = torch.norm(e2_vec - e3_vec).item()
                euc_proximity = 1 / (1 + euc_dist)

                max_cos_sim = max(max_cos_sim, cos_sim)
                max_euc_proximity = max(max_euc_proximity, euc_proximity)

        pair_results.append(
            {
                "e2": e2,
                "e3": e3,
                "max_cos_sim": max_cos_sim,
                "max_euc_proximity": max_euc_proximity,
                "no_cot_performance": row["mean"],
                "n_samples": row["count"],
            }
        )

    results_df = pd.DataFrame(pair_results)
    return results_df


def plot_pair_heatmap(pair_df: pd.DataFrame, min_samples: int = 1) -> plt.Figure:
    """Plot heatmap of pair similarities and performance."""
    # Filter for pairs with sufficient samples
    plot_df = pair_df[pair_df["n_samples"] >= min_samples].copy()

    # Create pair labels
    plot_df["pair"] = plot_df.apply(lambda x: f"{x['e2']} → {x['e3']}", axis=1)
    # Sort by max_cos_sim
    plot_df = plot_df.sort_values(by="max_cos_sim", ascending=False)
    # Create matrix for heatmap
    matrix_data = plot_df[["max_cos_sim", "max_euc_proximity", "no_cot_performance"]].values

    # Plot
    fig, ax = plt.subplots(figsize=(8, len(plot_df) * 0.3))
    sns.heatmap(
        matrix_data,
        ax=ax,
        cmap="RdYlBu",
        center=0,
        annot=True,
        fmt=".2f",
        yticklabels=plot_df["pair"],
        xticklabels=["Max Cosine Sim", "Max Euclidean Proximity", "Avg Performance"],
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig


# After loading activations and triplets:
pair_df = analyze_pair_similarities(f"{model_name_pathsafe}", triplets_df)

fig = plot_pair_heatmap(pair_df, min_samples=1)
plt.show()


# %%
def analyze_type_pair_similarities(
    activations_path: str,
    triplets_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze similarities grouped by (e2_type, e3_type) pairs.
    Returns a DataFrame with average similarities and performance for each type pair.
    """
    # Load activations for e2 and e3
    e2_activations = {
        entity: torch.load(f"{activations_path}/{entity_to_filename(entity)}.pt")
        for entity in tqdm(triplets_df["e2_label"].unique())
    }
    e3_activations = {
        entity: torch.load(f"{activations_path}/{entity_to_filename(entity)}.pt")
        for entity in tqdm(triplets_df["e3_label"].unique())
    }

    # Get all layer names
    layer_names = list(next(iter(e2_activations.values())).keys())

    # Store results for each pair
    pair_results = []

    # Group by type pairs and compute average performance ratio
    type_performance = (
        triplets_df.groupby(["e2_type", "e3_type"])
        .agg(
            {
                "two_hop_no_cot": "mean",
                "two_hop_cot": "mean",  # Adding CoT performance
                "e2_label": "count",  # For sample count
            }
        )
        .reset_index()
    )
    # Calculate performance ratio
    # Add small epsilon to denominator to avoid division by zero
    type_performance["performance_ratio"] = type_performance["two_hop_no_cot"] / (
        type_performance["two_hop_cot"] + 1e-8
    )

    # For each type pair
    for _, row in tqdm(type_performance.iterrows(), total=len(type_performance), desc="Type pairs"):
        e2_type, e3_type = row["e2_type"], row["e3_type"]

        # Get all entities of these types
        type_pairs_df = triplets_df[
            (triplets_df["e2_type"] == e2_type) & (triplets_df["e3_type"] == e3_type)
        ]

        cos_sims = []
        euc_proximities = []

        # For each entity pair of these types
        for _, pair_row in type_pairs_df.iterrows():
            e2, e3 = pair_row["e2_label"], pair_row["e3_label"]

            # Skip if we don't have activations
            if e2 not in e2_activations or e3 not in e3_activations:
                continue

            # Find max similarities across all layer combinations
            max_cos_sim = float("-inf")
            max_euc_proximity = float("-inf")

            for src_layer in layer_names:
                for tgt_layer in layer_names:
                    e2_vec = e2_activations[e2][src_layer]
                    e3_vec = e3_activations[e3][tgt_layer]

                    cos_sim = torch.nn.functional.cosine_similarity(
                        e2_vec.unsqueeze(0), e3_vec.unsqueeze(0)
                    ).item()
                    euc_dist = torch.norm(e2_vec - e3_vec).item()
                    euc_proximity = 1 / (1 + euc_dist)

                    max_cos_sim = max(max_cos_sim, cos_sim)
                    max_euc_proximity = max(max_euc_proximity, euc_proximity)

            cos_sims.append(max_cos_sim)
            euc_proximities.append(max_euc_proximity)

        if cos_sims:  # if we have valid pairs
            pair_results.append(
                {
                    "e2_type": e2_type,
                    "e3_type": e3_type,
                    "avg_cos_sim": np.mean(cos_sims),
                    "avg_euc_proximity": np.mean(euc_proximities),
                    "performance_ratio": row["performance_ratio"],
                    "n_samples": row["e2_label"],
                }
            )

    results_df = pd.DataFrame(pair_results)
    return results_df


def plot_type_pair_heatmap(type_df: pd.DataFrame, min_samples: int = 1) -> plt.Figure:
    """Plot heatmap of type pair similarities and performance."""
    # Filter for pairs with sufficient samples
    plot_df = type_df[type_df["n_samples"] >= min_samples].copy()

    # Create type pair labels
    plot_df["type_pair"] = plot_df.apply(lambda x: f"{x['e2_type']} → {x['e3_type']}", axis=1)
    # Sort by avg_cos_sim
    plot_df = plot_df.sort_values(by="avg_cos_sim", ascending=False)

    # Create matrix for heatmap
    matrix_data = plot_df[["avg_cos_sim", "avg_euc_proximity", "performance_ratio"]].values

    # Plot
    fig, ax = plt.subplots(figsize=(8, len(plot_df) * 0.3))
    sns.heatmap(
        matrix_data,
        ax=ax,
        cmap="RdYlBu",
        center=0,
        annot=True,
        fmt=".2f",
        yticklabels=plot_df["type_pair"],
        xticklabels=[
            "Avg Max Cosine Sim",
            "Avg Max Euclidean Proximity",
            "Performance Ratio (no-CoT/CoT)",
        ],
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig


# Run the analysis:
# type_df = analyze_type_pair_similarities(
#     torch.load("e2/activations.pt"),
#     torch.load("e3/activations.pt"),
#     triplets_df
# )
# Correlation:
# After creating type_df, add:

# Compute correlations
correlations = pd.DataFrame(
    {
        "Cosine Sim vs Performance": [
            type_df["avg_cos_sim"].corr(type_df["performance_ratio"]),
            type_df["avg_cos_sim"].corr(type_df["performance_ratio"], method="spearman"),
        ],
        "Euclidean Proximity vs Performance": [
            type_df["avg_euc_proximity"].corr(type_df["performance_ratio"]),
            type_df["avg_euc_proximity"].corr(type_df["performance_ratio"], method="spearman"),
        ],
    },
    index=["Pearson", "Spearman"],
)

print("\nCorrelations with Performance Ratio (no-CoT/CoT):")
print(correlations.round(3))

# Optional: Create scatter plots to visualize relationships
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Cosine similarity scatter
sns.scatterplot(data=type_df, x="avg_cos_sim", y="performance_ratio", ax=ax1)
ax1.set_title("Cosine Similarity vs Performance Ratio")
ax1.set_xlabel("Average Max Cosine Similarity")
ax1.set_ylabel("Performance Ratio (no-CoT/CoT)")

# Euclidean proximity scatter
sns.scatterplot(data=type_df, x="avg_euc_proximity", y="performance_ratio", ax=ax2)
ax2.set_title("Euclidean Proximity vs Performance Ratio")
ax2.set_xlabel("Average Max Euclidean Proximity")
ax2.set_ylabel("Performance Ratio (no-CoT/CoT)")

plt.tight_layout()
plt.show()

fig = plot_type_pair_heatmap(type_df, min_samples=2)
plt.show()

# %%

# drop rows with NaN or inf values
type_df.dropna(inplace=True)
type_df = type_df[type_df["performance_ratio"] != np.inf]
type_df


# %%

triplets_df.groupby(["e2_type", "e3_type"])["r2_template"].value_counts()
