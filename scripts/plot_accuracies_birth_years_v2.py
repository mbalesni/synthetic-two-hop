# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from tqdm import tqdm

# List of attributes to plot
ATTRIBUTES = [
    "nobel_literature",
    "g7_summit_city",
    "g7_host_country",
    "eurovision_host",
    "oscar_best_picture",
    "oscar_best_director",
    "grammy_album",
    "nba_champion",
    "f1_champion",
    "time_person",
    "odd_even",
    "leap_year",
]

# Single experiment config for birth years
EXPERIMENT_CONFIG = {
    "experiments/semi_synthetic/jiahai_birth_years.yaml": {
        "label": "Birth Years",
        "color": sns.color_palette("tab10")[0],
    }
}


def get_accuracy_metrics_for_runs():
    """Get accuracy metrics from W&B using server-side filtering."""
    api = wandb.Api()

    # Create filter for the birth years experiment
    filters = {
        "$or": [{"config.experiment_config_path": path} for path in EXPERIMENT_CONFIG.keys()]
    }

    # Fetch runs
    runs = api.runs(
        "sita/latent_reasoning",
        filters=filters,
        order="+created_at",
    )

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for run in tqdm(runs, desc="Processing W&B runs", total=len(runs)):
        config_path = run.config.get("experiment_config_path", "")
        if not config_path:
            continue

        for attr in ATTRIBUTES:
            nocot_metric = f"acc_{attr}_nocot"
            cot_metric = f"acc_{attr}_cot"

            if (
                run.summary.get(nocot_metric, None) is None
                or run.summary.get(cot_metric, None) is None
            ):
                continue

            nocot_data = run.history(keys=["train/epoch", nocot_metric])
            cot_data = run.history(keys=["train/epoch", cot_metric])

            if not nocot_data.empty:
                data[attr]["nocot"][config_path].extend(nocot_data.values.tolist())
            if not cot_data.empty:
                data[attr]["cot"][config_path].extend(cot_data.values.tolist())

    return data


def compute_statistics(values):
    """Compute mean and standard error for each epoch."""
    if not values:
        return np.array([]), np.array([]), np.array([])

    values = np.array(values)
    epochs = values[:, 1]
    measurements = values[:, 2]

    unique_epochs = np.unique(epochs)
    means = []
    sems = []

    for epoch in unique_epochs:
        epoch_values = measurements[epochs == epoch]
        means.append(np.mean(epoch_values))
        sems.append(np.std(epoch_values) / np.sqrt(len(epoch_values)))

    return unique_epochs, np.array(means), np.array(sems)


def plot_accuracies(data: dict, ylim: float = 1.0):
    """Create plots comparing CoT and no-CoT accuracies."""
    print("Creating plots...")
    
    # Set up the plot grid (4x3 for 12 attributes)
    n_rows, n_cols = 4, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    axes = axes.flatten()

    # Plot each attribute
    for idx, attr in enumerate(ATTRIBUTES):
        ax = axes[idx]

        # Plot each experimental condition
        for config_path, config_info in EXPERIMENT_CONFIG.items():
            # Use different colors for no-CoT and CoT
            nocot_color = sns.color_palette("tab10")[0]  # Blue
            cot_color = sns.color_palette("tab10")[1]    # Orange

            # Plot no-CoT accuracy
            if config_path in data[attr]["nocot"]:
                values = data[attr]["nocot"][config_path]
                epochs, means, sems = compute_statistics(values)
                if len(epochs) > 0:
                    ax.plot(epochs, means, color=nocot_color, label="No-CoT")
                    ax.fill_between(epochs, means - sems, means + sems, color=nocot_color, alpha=0.2)

            # Plot CoT accuracy
            if config_path in data[attr]["cot"]:
                values = data[attr]["cot"][config_path]
                epochs, means, sems = compute_statistics(values)
                if len(epochs) > 0:
                    ax.plot(epochs, means, color=cot_color, label="CoT")
                    ax.fill_between(epochs, means - sems, means + sems, color=cot_color, alpha=0.2)

        # Customize subplot
        ax.set_title(attr.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, ylim)

        # Add legend to first subplot only
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Remove any empty subplots
    for idx in range(len(ATTRIBUTES), len(axes)):
        fig.delaxes(axes[idx])

    print("Saving plot...")
    plt.tight_layout()
    plt.savefig("birth_years_accuracy_plots.png", bbox_inches="tight", dpi=300)
    plt.show()


# %%
print("Fetching accuracy data from W&B...")
data = get_accuracy_metrics_for_runs()

# %%
# Plot with default y-limit (1.0)
plot_accuracies(data)

# Example with custom y-limit:
# plot_accuracies(data, ylim=0.5) 