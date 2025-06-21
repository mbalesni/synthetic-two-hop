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


def get_metrics_for_runs():
    """Get metrics from W&B using server-side filtering."""
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
            normal_metric = f"eval/{attr}_nocot_loss"
            shuffled_metric = f"eval/{attr}_shuffled_loss"

            if (
                run.summary.get(normal_metric, None) is None
                or run.summary.get(shuffled_metric, None) is None
            ):
                continue

            normal_data = run.history(keys=["train/epoch", normal_metric])
            shuffled_data = run.history(keys=["train/epoch", shuffled_metric])

            if not normal_data.empty:
                data[attr]["normal"][config_path].extend(normal_data.values.tolist())
            if not shuffled_data.empty:
                data[attr]["shuffled"][config_path].extend(shuffled_data.values.tolist())

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


def plot_losses(data: dict, ylims: dict = None):
    """Create plots of losses with optional custom y-axis limits."""
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
            color = config_info["color"]
            label = config_info["label"]

            # Plot normal loss
            if config_path in data[attr]["normal"]:
                values = data[attr]["normal"][config_path]
                epochs, means, sems = compute_statistics(values)
                if len(epochs) > 0:
                    ax.plot(epochs, means, color=color, label=label)
                    ax.fill_between(epochs, means - sems, means + sems, color=color, alpha=0.2)

            # Plot shuffled loss (dashed)
            if config_path in data[attr]["shuffled"]:
                values = data[attr]["shuffled"][config_path]
                epochs, means, sems = compute_statistics(values)
                if len(epochs) > 0:
                    ax.plot(epochs, means, color=color, linestyle="--", label=f"{label} (Shuffled)")
                    ax.fill_between(epochs, means - sems, means + sems, color=color, alpha=0.2)

        # Customize subplot
        ax.set_title(attr.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        # Set y-axis limit if specified for this attribute
        if ylims and attr in ylims:
            ax.set_ylim(top=ylims[attr])

        # Add legend to first subplot only
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Remove any empty subplots
    for idx in range(len(ATTRIBUTES), len(axes)):
        fig.delaxes(axes[idx])

    print("Saving plot...")
    plt.tight_layout()
    plt.savefig("birth_years_loss_plots.png", bbox_inches="tight", dpi=300)
    plt.show()


# %%
print("Fetching data from W&B...")
data = get_metrics_for_runs()

# %%
# Plot with auto scaling
plot_losses(data)

# Example of how to use custom y-limits if needed:
# custom_ylims = {
#     "nobel_literature": 4.0,
#     "g7_summit_city": 3.5,
#     # ... add more as needed
# }
# plot_losses(data, ylims=custom_ylims)
