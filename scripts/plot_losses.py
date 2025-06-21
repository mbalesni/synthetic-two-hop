# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from tqdm import tqdm

# List of attributes to plot
ATTRIBUTES = [
    "capital",
    "currency",
    "tld",
    "calling_code",
    "flag_color",
    "language",
    "national_animal",
    "largest_stadium",
    "stock_exchange",
    "national_flower",
    "largest_airport",
]

# Mapping of config paths to their descriptions and colors
EXPERIMENT_CONFIGS = {
    "experiments/semi_synthetic/jiahai_fictional_countries_11hops.yaml": {
        "label": "Original (everything related)",
        "color": sns.color_palette("tab10")[0],
    },
    "experiments/fully_synthetic_countries_shuffled/fictional_countries_shuffled_11hops.yaml": {
        "label": "Permuted Names",
        "color": sns.color_palette("tab10")[1],
    },
    "experiments/semi_synthetic/jiahai_fictional_countries_unrelated_11hops.yaml": {
        "label": "Permuted Everything (Poor filtering)",
        "color": sns.color_palette("tab10")[2],
    },
    "experiments/fully_synthetic_countries_v3/fictional_countries_v3_11hops.yaml": {
        "label": "Permuted Everything (Good filtering)",
        "color": sns.color_palette("tab10")[3],
    },
}


def get_metrics_for_runs():
    """Get metrics from W&B using server-side filtering."""
    api = wandb.Api()

    # Create filters for the specific experiments we want
    filters = {
        "$or": [{"config.experiment_config_path": path} for path in EXPERIMENT_CONFIGS.keys()]
    }

    # Fetch runs with specific metrics and configs
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
            shuffled_metric = f"eval/{attr}_nocot_shuffled_loss"

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

    # Convert to numpy array for faster processing
    values = np.array(values)
    epochs = values[:, 1]
    measurements = values[:, 2]

    # Get unique epochs
    unique_epochs = np.unique(epochs)

    means = []
    sems = []

    for epoch in unique_epochs:
        epoch_values = measurements[epochs == epoch]
        means.append(np.mean(epoch_values))
        sems.append(np.std(epoch_values) / np.sqrt(len(epoch_values)))

    return unique_epochs, np.array(means), np.array(sems)


def plot_losses(data: dict):
    """Create plots of losses."""

    print("Creating plots...")
    # Set up the plot grid
    n_rows, n_cols = 4, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    axes = axes.flatten()

    # Y-axis limits for specific plots
    ylims = {
        "currency": 4.5,
        "flag_color": 2.0,
        "national_animal": 3.5,
        "largest_stadium": 2.75,
        "stock_exchange": 5.25,
        "national_flower": 3.25,
        "largest_airport": 3.5,
        "capital": 6.25,
    }

    # Plot each attribute
    for idx, attr in enumerate(ATTRIBUTES):
        ax = axes[idx]

        # Plot each experimental condition
        for config_path, config_info in EXPERIMENT_CONFIGS.items():
            color = config_info["color"]
            label = config_info["label"]

            # Plot normal loss
            if config_path in data[attr]["normal"]:
                values = data[attr]["normal"][config_path]
                epochs, means, sems = compute_statistics(values)
                if len(epochs) > 0:  # Only plot if we have data
                    ax.plot(epochs, means, color=color, label=label)
                    ax.fill_between(epochs, means - sems, means + sems, color=color, alpha=0.2)

            # Plot shuffled loss (dashed)
            if config_path in data[attr]["shuffled"]:
                values = data[attr]["shuffled"][config_path]
                epochs, means, sems = compute_statistics(values)
                if len(epochs) > 0:  # Only plot if we have data
                    ax.plot(epochs, means, color=color, linestyle="--")
                    ax.fill_between(epochs, means - sems, means + sems, color=color, alpha=0.2)

        # Customize subplot
        ax.set_title(attr.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        # Set y-axis limit if specified for this attribute
        if attr in ylims:
            ax.set_ylim(top=ylims[attr])

        # Add legend to first subplot only
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Remove any empty subplots
    for idx in range(len(ATTRIBUTES), len(axes)):
        fig.delaxes(axes[idx])

    print("Saving plot...")
    plt.tight_layout()
    plt.savefig("loss_plots.png", bbox_inches="tight", dpi=300)
    plt.show()


def get_accuracy_metrics_for_runs(metric_name: str = "acc_{attr}_cot"):
    """Get accuracy metrics from W&B using server-side filtering."""
    api = wandb.Api()

    filters = {
        "$or": [{"config.experiment_config_path": path} for path in EXPERIMENT_CONFIGS.keys()]
    }

    runs = api.runs(
        "sita/latent_reasoning",
        filters=filters,
        order="+created_at",
    )

    data = defaultdict(lambda: defaultdict(list))

    for run in tqdm(runs, desc="Processing W&B runs", total=len(runs)):
        config_path = run.config.get("experiment_config_path", "")
        if not config_path:
            continue

        for attr in ATTRIBUTES:
            metric = metric_name.format(attr=attr)

            if run.summary.get(metric, None) is None:
                continue

            metric_data = run.history(keys=["train/epoch", metric])
            if not metric_data.empty:
                data[attr][config_path].extend(metric_data.values.tolist())

    return data


def plot_accuracies(data: dict, metric_name: str, ylim: float = 1.0):
    """Create plots of CoT accuracies."""
    print("Creating accuracy plots...")
    n_rows, n_cols = 4, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    axes = axes.flatten()

    for idx, attr in enumerate(ATTRIBUTES):
        ax = axes[idx]

        for config_path, config_info in EXPERIMENT_CONFIGS.items():
            color = config_info["color"]
            label = config_info["label"]

            if config_path in data[attr]:
                values = data[attr][config_path]
                epochs, means, sems = compute_statistics(values)
                if len(epochs) > 0:
                    ax.plot(epochs, means, color=color, label=label)
                    ax.fill_between(epochs, means - sems, means + sems, color=color, alpha=0.2)

        ax.set_title(f"{attr.replace('_', ' ').title()} {metric_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, ylim)  # Accuracy is between 0 and 1

        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    for idx in range(len(ATTRIBUTES), len(axes)):
        fig.delaxes(axes[idx])

    print("Saving accuracy plot...")
    plt.tight_layout()
    plt.savefig("accuracy_plots.png", bbox_inches="tight", dpi=300)
    plt.show()


# %%
print("Fetching data from W&B...")
data = get_metrics_for_runs()

# %%

plot_losses(data)
# %%

print("Fetching CoT accuracy data from W&B...")
cot_accuracy_data = get_accuracy_metrics_for_runs(metric_name="acc_{attr}_cot")

# %%
print("Fetching NoCoT accuracy data from W&B...")
nocot_accuracy_data = get_accuracy_metrics_for_runs(metric_name="acc_{attr}_nocot")
# %%
plot_accuracies(cot_accuracy_data, metric_name="CoT Accuracy")
# %%
plot_accuracies(nocot_accuracy_data, metric_name="No-CoT Accuracy", ylim=0.3)
# %%
