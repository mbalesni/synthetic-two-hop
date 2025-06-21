# %%
import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from numpy.typing import NDArray
from scipy import stats
from tqdm import tqdm

# Dictionary mapping experiment config paths to their display info
# Get colors from tab20 palette for 17 datasets
colors = sns.color_palette("tab20", n_colors=20)


def apply_plot_style(ax, remove_grid=True, spine_width=1.5, tick_size=20):
    """Apply consistent plot styling to match paper figures.

    Args:
        ax: Matplotlib axis object
        remove_grid: Whether to remove grid lines
        spine_width: Width of the axis spines
        tick_size: Font size for tick labels
    """
    # Remove grid
    if remove_grid:
        ax.grid(False)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Make remaining spines thicker
    ax.spines["left"].set_linewidth(spine_width)
    ax.spines["bottom"].set_linewidth(spine_width)

    # Increase tick width
    ax.tick_params(width=spine_width, length=6)

    # Set tick label size
    ax.tick_params(axis="both", labelsize=tick_size)


EXPERIMENT_CONFIG = {
    "experiments/semi_synthetic/configs/parks.yaml": {
        "label": "Parks",
        "color": colors[0],
        "attributes": ["state", "established", "code"],
    },
    "experiments/semi_synthetic/configs/chemical_elements.yaml": {
        "label": "Chemical Elements",
        "color": colors[1],
        "attributes": ["atomic_number", "symbol", "discovery_year", "discoverer_last_name"],
    },
    "experiments/semi_synthetic/configs/programming_languages.yaml": {
        "label": "Programming Languages",
        "color": colors[2],
        "attributes": ["release_year", "file_extension", "creator_last_name"],
    },
    "experiments/semi_synthetic/configs/world_heritage_sites.yaml": {
        "label": "World Heritage Sites",
        "color": colors[3],
        "attributes": ["year_inscribed", "country", "city", "continent"],
    },
    "experiments/semi_synthetic/configs/video_game_consoles.yaml": {
        "label": "Video Game Consoles",
        "color": colors[4],
        "attributes": ["release_year", "manufacturer", "home_country", "generation"],
    },
    "experiments/semi_synthetic/configs/famous_paintings.yaml": {
        "label": "Famous Paintings",
        "color": colors[5],
        "attributes": ["creation_year", "artist_last_name", "museum", "city"],
    },
    "experiments/semi_synthetic/configs/cathedrals.yaml": {
        "label": "Cathedrals",
        "color": colors[6],
        "attributes": ["completion_year", "city", "architectural_style", "country"],
    },
    "experiments/semi_synthetic/configs/bridges.yaml": {
        "label": "Bridges",
        "color": colors[7],
        "attributes": ["completion_year", "city", "country"],
    },
    "experiments/semi_synthetic/configs/operas.yaml": {
        "label": "Operas",
        "color": colors[8],
        "attributes": ["premiere_year", "composer_last_name", "language", "premiere_city"],
    },
    "experiments/semi_synthetic/configs/telescopes.yaml": {
        "label": "Telescopes",
        "color": colors[9],
        "attributes": ["first_light_year", "location", "country", "continent"],
    },
    "experiments/semi_synthetic/configs/ancient_cities.yaml": {
        "label": "Ancient Cities",
        "color": colors[10],
        "attributes": ["founding_century_bc", "modern_country", "continent"],
    },
    "experiments/semi_synthetic/configs/mountain_peaks.yaml": {
        "label": "Mountain Peaks",
        "color": colors[11],
        "attributes": ["first_ascent_year", "country", "continent", "height_meters"],
    },
    "experiments/semi_synthetic/configs/universities.yaml": {
        "label": "Universities",
        "color": colors[12],
        "attributes": ["founding_year", "city", "country", "continent"],
    },
    "experiments/semi_synthetic/configs/constellations.yaml": {
        "label": "Constellations",
        "color": colors[13],
        "attributes": ["brightest_star", "best_viewing_month", "hemisphere"],
    },
    "experiments/semi_synthetic/configs/ships.yaml": {
        "label": "Ships",
        "color": colors[14],
        "attributes": ["launch_year", "first_captain_last_name", "home_port", "country"],
    },
    "experiments/semi_synthetic/configs/newspapers.yaml": {
        "label": "Newspapers",
        "color": colors[15],
        "attributes": ["founding_year", "language", "city", "country"],
    },
    "experiments/semi_synthetic/configs/subway_systems.yaml": {
        "label": "Subway Systems",
        "color": colors[16],
        "attributes": ["opening_year", "station_count", "city", "country"],
    },
}
base_path = "latent_reasoning"
EXPERIMENT_CONFIG = {f"{base_path}/{path}": config for path, config in EXPERIMENT_CONFIG.items()}

# Mapping of individual e3_types (attributes) to meta-categories
# This groups conceptually similar attributes across different datasets
E3_META_CATEGORIES = {
    # Location-based attributes
    "location_city": [
        "city",
        "premiere_city",
        "home_port",
    ],
    "location_country": [
        "country",
        "modern_country",
        "home_country",
        "registry_country",
    ],
    "location_continent": [
        "continent",
    ],
    "location_state": [
        "state",
        "state_province",
    ],
    "location_specific": [
        "location",  # For telescopes
    ],
    # All year-based attributes
    "year": [
        # Establishment years
        "established",
        "founding_year",
        "founding_year_bc",
        "opening_year",
        "operational_year",
        # Creation years
        "creation_year",
        "year_composed",
        "premiere_year",
        "release_year",
        "launch_year",
        "completion_year",
        "construction_year",
        "year_inscribed",
        "first_light_year",
        "first_ascent_year",
        # Discovery years
        "discovery_year",
        "year_recognized",
    ],
    # Person names
    "person_last_name": [
        "artist_last_name",
        "creator_last_name",
        "composer_last_name",
        "discoverer_last_name",
        "first_captain_last_name",
        "founder_last_name",
        "designer_last_name",
        "architect_last_name",
        "first_ascender_last_name",
    ],
    # Numeric properties
    "atomic_number": [
        "atomic_number",
    ],
    "numeric_count": [
        "station_count",
        "terminal_count",
        "line_count",
    ],
    "console_generation": [
        "generation",
    ],
    "numeric_measurement": [
        "height_meters",
        "altitude",
        "altitude_meters",
        "mirror_size",
        "track_gauge",
    ],
    # Language and communication
    "language": [
        "language",
    ],
    # Identifiers and codes
    "park_code": [
        "code",  # Park codes
    ],
    "chemical_element_symbol": [
        "symbol",  # Chemical element symbols
    ],
    "file_extension": [
        "file_extension",  # Programming language extensions
    ],
    # Organizations and institutions
    "manufacturer": [
        "manufacturer",  # Video game console manufacturers
    ],
    "museum": [
        "museum",  # Where paintings are housed
    ],
    # Style and type categories
    "architectural_style": [
        "architectural_style",  # Cathedral styles
    ],
    # Natural/astronomical features
    "star": [
        "brightest_star",  # In constellations
    ],
    "hemisphere": [
        "hemisphere",  # Where constellations are visible
    ],
    # Time periods (non-year)
    "month": [
        "best_viewing_month",  # For constellations
    ],
    "century": [
        "founding_century_bc",  # Century is still temporal
    ],
}

# Create reverse mapping for easy lookup
E3_TO_META = {}
for meta_category, e3_list in E3_META_CATEGORIES.items():
    for e3 in e3_list:
        E3_TO_META[e3] = meta_category


def get_metrics_for_runs() -> DefaultDict[
    str, DefaultDict[str, DefaultDict[str, DefaultDict[str, List[List[float]]]]]
]:
    """Get metrics from W&B using server-side filtering."""
    api = wandb.Api()
    # Create filter for the january push experiments
    filters = {
        "$or": [{"config.experiment_config_path": path} for path in EXPERIMENT_CONFIG.keys()]
    }

    # Fetch runs
    runs = api.runs(
        "sita/latent_reasoning",
        filters=filters,
        order="+created_at",
    )

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for run in tqdm(runs, desc="Processing W&B runs", total=len(runs)):
        config_path = run.config.get("experiment_config_path", "")
        if not config_path or config_path not in EXPERIMENT_CONFIG:
            print(f"Skipping run {run.id} with config path {config_path}")
            continue

        # Get attributes for this experiment
        attributes = EXPERIMENT_CONFIG[config_path]["attributes"]

        for attr in attributes:
            # Loss metrics
            normal_metric = f"eval/2hop_{attr}_nocot_loss"
            shuffled_metric = f"eval/2hop_{attr}_nocot_shuffled_loss"
            acc_metric = f"acc_2hop_{attr}_nocot"
            acc_cot_metric = f"acc_2hop_{attr}_cot"
            acc_a_metric = f"acc_a"  # First hop accuracy

            if (
                run.summary.get(normal_metric, None) is None
                or run.summary.get(shuffled_metric, None) is None
                or run.summary.get(acc_metric, None) is None
            ):
                print(f"Skipping {attr} from run {run.id} with config path {config_path}")
                print(f"{run.summary.get(normal_metric, None)}")
                print(f"{run.summary.get(shuffled_metric, None)}")
                print(f"{run.summary.get(acc_metric, None)}")
                print("--------------------------------")
                continue

            normal_data = run.history(keys=["train/epoch", normal_metric])
            shuffled_data = run.history(keys=["train/epoch", shuffled_metric])
            acc_data = run.history(keys=["train/epoch", acc_metric])
            acc_cot_data = run.history(keys=["train/epoch", acc_cot_metric])
            acc_a_data = run.history(keys=["train/epoch", acc_a_metric])

            if not normal_data.empty:
                data[config_path][attr]["normal"][config_path].extend(normal_data.values.tolist())
            if not shuffled_data.empty:
                data[config_path][attr]["shuffled"][config_path].extend(
                    shuffled_data.values.tolist()
                )
            if not acc_data.empty:
                data[config_path][attr]["accuracy"][config_path].extend(acc_data.values.tolist())
            if not acc_cot_data.empty:
                data[config_path][attr]["accuracy_cot"][config_path].extend(
                    acc_cot_data.values.tolist()
                )
            if not acc_a_data.empty:
                data[config_path][attr]["accuracy_a"][config_path].extend(
                    acc_a_data.values.tolist()
                )

    return data


def compute_statistics(
    values: List[List[float]],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute mean and standard error for each epoch."""
    if not values:
        return np.array([]), np.array([]), np.array([])

    values_array = np.array(values)
    epochs = values_array[:, 1]
    measurements = values_array[:, 2]

    unique_epochs = np.unique(epochs)
    means = []
    sems = []

    for epoch in unique_epochs:
        epoch_values = measurements[epochs == epoch]
        means.append(np.mean(epoch_values))
        sems.append(np.std(epoch_values) / np.sqrt(len(epoch_values)))

    return unique_epochs, np.array(means), np.array(sems)


def plot_losses(
    data: Mapping[str, Mapping[str, Mapping[str, Mapping[str, List[List[float]]]]]],
    ylims: Optional[Dict[str, float]] = None,
) -> None:
    """Create plots of losses with optional custom y-axis limits."""
    print("Creating plots...")

    # Calculate total number of datasets and max attributes per dataset
    n_datasets = len(EXPERIMENT_CONFIG)
    max_attributes = max(len(config["attributes"]) for config in EXPERIMENT_CONFIG.values())

    # Set up the plot grid
    fig = plt.figure(figsize=(5 * max_attributes, 4 * n_datasets))

    # Plot each dataset and its attributes
    for dataset_idx, (config_path, config_info) in enumerate(EXPERIMENT_CONFIG.items()):
        label = config_info["label"]
        color = config_info["color"]
        attributes = config_info["attributes"]

        # Create subplot for each attribute
        for attr_idx, attr in enumerate(attributes):
            ax = plt.subplot(
                n_datasets, max_attributes, dataset_idx * max_attributes + attr_idx + 1
            )

            # Plot normal loss
            if attr in data[config_path] and "normal" in data[config_path][attr]:
                values = data[config_path][attr]["normal"][config_path]
                epochs, means, sems = compute_statistics(values)
                if len(epochs) > 0:
                    ax.plot(epochs, means, color=color, label=label, linewidth=2.5)
                    ax.fill_between(epochs, means - sems, means + sems, color=color, alpha=0.2)

            # Plot shuffled loss (dashed)
            if attr in data[config_path] and "shuffled" in data[config_path][attr]:
                values = data[config_path][attr]["shuffled"][config_path]
                epochs, means, sems = compute_statistics(values)
                if len(epochs) > 0:
                    ax.plot(
                        epochs,
                        means,
                        color=color,
                        linestyle="--",
                        label=f"{label} (Shuffled)",
                        linewidth=2.5,
                    )
                    ax.fill_between(epochs, means - sems, means + sems, color=color, alpha=0.2)

            # Customize subplot
            ax.set_title(f"{label}: {attr.replace('_', ' ').title()}", pad=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")

            # Apply consistent styling
            apply_plot_style(ax, remove_grid=True, spine_width=1.5, tick_size=12)

            # Add legend to first subplot only
            if dataset_idx == 0 and attr_idx == 0:
                ax.legend()

            # Set y-axis limit if specified
            if ylims and f"{config_path}_{attr}" in ylims:
                ax.set_ylim(top=ylims[f"{config_path}_{attr}"])

    print("Saving plot...")
    plt.tight_layout()
    plt.savefig("january_push_loss_plots.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_accuracies(
    data: Mapping[str, Mapping[str, Mapping[str, Mapping[str, List[List[float]]]]]],
    ylims: Optional[Dict[str, float]] = None,
    metric_type: str = "nocot",  # "nocot" or "cot"
) -> None:
    """Create plots of accuracies."""
    print(f"Creating accuracy plots (metric_type: {metric_type})...")

    # Calculate total number of datasets and max attributes per dataset
    n_datasets = len(EXPERIMENT_CONFIG)
    max_attributes = max(len(config["attributes"]) for config in EXPERIMENT_CONFIG.values())

    # Set up the plot grid
    fig = plt.figure(figsize=(5 * max_attributes, 4 * n_datasets))

    # Plot each dataset and its attributes
    for dataset_idx, (config_path, config_info) in enumerate(EXPERIMENT_CONFIG.items()):
        label = config_info["label"]
        color = config_info["color"]
        attributes = config_info["attributes"]

        # Create subplot for each attribute
        for attr_idx, attr in enumerate(attributes):
            ax = plt.subplot(
                n_datasets, max_attributes, dataset_idx * max_attributes + attr_idx + 1
            )

            # Get accuracy data with the appropriate metric type
            accuracy_key = "accuracy_cot" if metric_type == "cot" else "accuracy"
            if attr in data[config_path] and accuracy_key in data[config_path][attr]:
                values = data[config_path][attr][accuracy_key][config_path]
                epochs, means, sems = compute_statistics(values)
                if len(epochs) > 0:
                    ax.plot(epochs, means, color=color, label=label, linewidth=3)
                    ax.fill_between(epochs, means - sems, means + sems, color=color, alpha=0.2)

            # Customize subplot
            title_suffix = " (CoT)" if metric_type == "cot" else " (No-CoT)"
            ax.set_title(f"{label}: {attr.replace('_', ' ').title()}{title_suffix}", pad=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")

            # Apply consistent styling
            apply_plot_style(ax, remove_grid=True, spine_width=1.5, tick_size=12)

            # Add legend to first subplot only
            if dataset_idx == 0 and attr_idx == 0:
                ax.legend()

            # Set y-axis limit if specified
            if ylims and f"{config_path}_{attr}" in ylims:
                ax.set_ylim(top=ylims[f"{config_path}_{attr}"])

    print("Saving accuracy plot...")
    plt.tight_layout()
    output_filename = f"semi_synthetic/{metric_type}_accuracy_by_e2_type.pdf"
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    plt.show()


def plot_averaged_losses(
    data: Mapping[str, Mapping[str, Mapping[str, Mapping[str, List[List[float]]]]]],
    ylim: Optional[float] = None,
    mode: str = "raw",  # "raw" or "difference"
    aggregate_attributes: bool = True,  # True to average across attributes, False to plot separately
    fig_height: Optional[float] = None,  # Custom figure height
) -> None:
    """Create plots of losses averaged across attributes for each dataset.

    Args:
        data: The data dictionary from W&B
        ylim: Optional y-axis limit
        mode: "raw" to plot raw losses, "difference" to plot (shuffled - non_shuffled)
        aggregate_attributes: If True, average across attributes. If False, plot each attribute separately
        fig_height: Custom figure height. If None, uses default (8 for aggregated, 16 for separated)
    """
    print(
        f"Creating averaged loss plots (mode: {mode}, aggregate_attributes: {aggregate_attributes})..."
    )

    # Set up the plot with appropriate height
    if fig_height is None:
        fig_height = (
            8 if aggregate_attributes else 16
        )  # Double height when showing attributes separately

    n_datasets = len(EXPERIMENT_CONFIG)
    fig, ax = plt.subplots(1, 1, figsize=(12, fig_height))

    # Define line styles for different attributes
    line_styles = [
        "-",
        "--",
        "-.",
        ":",
        (0, (3, 1, 1, 1)),
        (0, (5, 1)),
        (0, (3, 5, 1, 5)),
        (0, (1, 1)),
    ]

    # Store data for plotting
    dataset_labels = []
    normal_means_by_dataset = []
    normal_sems_by_dataset = []
    shuffled_means_by_dataset = []
    shuffled_sems_by_dataset = []

    # Process each dataset
    for dataset_idx, (config_path, config_info) in enumerate(EXPERIMENT_CONFIG.items()):
        label = config_info["label"]
        color = config_info["color"]
        attributes = config_info["attributes"]

        if aggregate_attributes:
            # Original behavior: collect and average all attributes
            normal_means_by_attr = []
            normal_sems_by_attr = []
            shuffled_means_by_attr = []
            shuffled_sems_by_attr = []
            epochs_list = []

            for attr in attributes:
                # Normal loss
                if attr in data[config_path] and "normal" in data[config_path][attr]:
                    values = data[config_path][attr]["normal"][config_path]
                    epochs, means, sems = compute_statistics(values)
                    if len(epochs) > 0:
                        normal_means_by_attr.append(means)
                        normal_sems_by_attr.append(sems)
                        epochs_list.append(epochs)

                # Shuffled loss
                if attr in data[config_path] and "shuffled" in data[config_path][attr]:
                    values = data[config_path][attr]["shuffled"][config_path]
                    epochs, means, sems = compute_statistics(values)
                    if len(epochs) > 0:
                        shuffled_means_by_attr.append(means)
                        shuffled_sems_by_attr.append(sems)

            # Average across attributes if we have data
            if (
                normal_means_by_attr
                and shuffled_means_by_attr
                and len(normal_means_by_attr) == len(shuffled_means_by_attr)
            ):
                # Use the first epochs array (they should all be the same)
                epochs = epochs_list[0]

                if mode == "difference":
                    # Calculate difference for each attribute first, then average
                    diff_means_by_attr = []
                    diff_sems_by_attr = []

                    for i in range(len(normal_means_by_attr)):
                        # Difference = shuffled - normal (positive when model is learning correctly)
                        diff_mean = shuffled_means_by_attr[i] - normal_means_by_attr[i]
                        # Propagate error: sqrt(sem_shuffled^2 + sem_normal^2)
                        diff_sem = np.sqrt(
                            shuffled_sems_by_attr[i] ** 2 + normal_sems_by_attr[i] ** 2
                        )

                        diff_means_by_attr.append(diff_mean)
                        diff_sems_by_attr.append(diff_sem)

                    # Average the differences across attributes
                    avg_diff_means = np.mean(diff_means_by_attr, axis=0)
                    # Propagate error for averaging
                    avg_diff_sems = np.sqrt(np.sum(np.square(diff_sems_by_attr), axis=0)) / np.sqrt(
                        len(diff_means_by_attr)
                    )

                    # Plot difference
                    ax.plot(
                        epochs,
                        avg_diff_means,
                        label=f"{label}",
                        color=color,
                        linewidth=3,
                    )
                    ax.fill_between(
                        epochs,
                        avg_diff_means - avg_diff_sems,
                        avg_diff_means + avg_diff_sems,
                        color=color,
                        alpha=0.2,
                    )

                else:  # mode == "raw"
                    # Average the means across attributes
                    avg_normal_means = np.mean(normal_means_by_attr, axis=0)
                    # Propagate error: sqrt(sum of squared sems) / sqrt(n_attributes)
                    avg_normal_sems = np.sqrt(
                        np.sum(np.square(normal_sems_by_attr), axis=0)
                    ) / np.sqrt(len(normal_means_by_attr))

                    # Plot normal loss
                    ax.plot(
                        epochs,
                        avg_normal_means,
                        label=f"{label}",
                        color=color,
                        linewidth=3,
                    )
                    ax.fill_between(
                        epochs,
                        avg_normal_means - avg_normal_sems,
                        avg_normal_means + avg_normal_sems,
                        color=color,
                        alpha=0.2,
                    )

                    # Average the means across attributes for shuffled
                    avg_shuffled_means = np.mean(shuffled_means_by_attr, axis=0)
                    # Propagate error
                    avg_shuffled_sems = np.sqrt(
                        np.sum(np.square(shuffled_sems_by_attr), axis=0)
                    ) / np.sqrt(len(shuffled_means_by_attr))

                    # Plot shuffled loss
                    ax.plot(
                        epochs,
                        avg_shuffled_means,
                        label=f"{label} (Shuffled)",
                        color=color,
                        linestyle="--",
                        linewidth=3,
                    )
                    ax.fill_between(
                        epochs,
                        avg_shuffled_means - avg_shuffled_sems,
                        avg_shuffled_means + avg_shuffled_sems,
                        color=color,
                        alpha=0.2,
                    )
        else:
            # New behavior: plot each attribute separately
            for attr_idx, attr in enumerate(attributes):
                # Use cycling line styles if we run out
                line_style = line_styles[attr_idx % len(line_styles)]

                if mode == "difference":
                    # Calculate difference for this attribute
                    if (
                        attr in data[config_path]
                        and "normal" in data[config_path][attr]
                        and "shuffled" in data[config_path][attr]
                    ):
                        normal_values = data[config_path][attr]["normal"][config_path]
                        shuffled_values = data[config_path][attr]["shuffled"][config_path]

                        normal_epochs, normal_means, normal_sems = compute_statistics(normal_values)
                        shuffled_epochs, shuffled_means, shuffled_sems = compute_statistics(
                            shuffled_values
                        )

                        if len(normal_epochs) > 0 and len(shuffled_epochs) > 0:
                            # Calculate difference
                            diff_means = shuffled_means - normal_means
                            # Propagate error
                            diff_sems = np.sqrt(shuffled_sems**2 + normal_sems**2)

                            # Plot with label only for first attribute
                            attr_label = f"{label} - {attr}" if attr_idx == 0 else f"    {attr}"
                            ax.plot(
                                normal_epochs,
                                diff_means,
                                label=attr_label,
                                color=color,
                                linestyle=line_style,
                                linewidth=2.5,
                            )
                            ax.fill_between(
                                normal_epochs,
                                diff_means - diff_sems,
                                diff_means + diff_sems,
                                color=color,
                                alpha=0.1,
                            )
                else:  # mode == "raw"
                    # Plot normal loss
                    if attr in data[config_path] and "normal" in data[config_path][attr]:
                        values = data[config_path][attr]["normal"][config_path]
                        epochs, means, sems = compute_statistics(values)
                        if len(epochs) > 0:
                            attr_label = f"{label} - {attr}" if attr_idx == 0 else f"    {attr}"
                            ax.plot(
                                epochs,
                                means,
                                label=attr_label,
                                color=color,
                                linestyle=line_style,
                                linewidth=2.5,
                            )
                            ax.fill_between(
                                epochs, means - sems, means + sems, color=color, alpha=0.1
                            )

                    # Plot shuffled loss (only in raw mode)
                    if attr in data[config_path] and "shuffled" in data[config_path][attr]:
                        values = data[config_path][attr]["shuffled"][config_path]
                        epochs, means, sems = compute_statistics(values)
                        if len(epochs) > 0:
                            # Don't add label for shuffled to avoid clutter
                            ax.plot(
                                epochs,
                                means,
                                color=color,
                                linestyle=line_style,
                                alpha=0.6,  # Make shuffled slightly transparent
                                linewidth=2,
                            )

    # Customize plot
    ax.set_xlabel("Epoch", fontsize=22)
    if mode == "difference":
        ax.set_ylabel("Loss Advantage over Random Baseline", fontsize=22)
        title = (
            "Loss Advantage over Random Baseline, by e2_type"
            if aggregate_attributes
            else "Loss Advantage over Random Baseline, by e2_type and e3_type"
        )
        ax.set_title(title, fontsize=24, pad=20)
        ax.axhline(
            y=0, color="black", linestyle="-", alpha=0.3, linewidth=1
        )  # Add reference line at 0
    else:
        ax.set_ylabel("Average Loss", fontsize=22)
        title = (
            "Average Loss, by e2_type" if aggregate_attributes else "Loss, by e2_type and e3_type"
        )
        ax.set_title(title, fontsize=24, pad=20)

    # Apply consistent styling
    apply_plot_style(ax, remove_grid=True, spine_width=2, tick_size=20)

    legend = ax.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12 if not aggregate_attributes else 16
    )
    # Add legend title
    legend_title = "e2_type" if aggregate_attributes else "e2_type - e3_type"
    legend.set_title(legend_title)
    legend.get_title().set_fontsize(14 if not aggregate_attributes else 18)
    legend.get_title().set_fontweight("bold")

    if ylim:
        ax.set_ylim(0, ylim)

    plt.tight_layout()
    if mode == "difference":
        suffix = "_by_e2_type" if aggregate_attributes else "_by_e3_type"
        plt.savefig(f"semi_synthetic/loss_advantage{suffix}.pdf", bbox_inches="tight", dpi=300)
    else:
        suffix = "_by_e2_type" if aggregate_attributes else "_by_e3_type"
        plt.savefig(f"semi_synthetic/loss{suffix}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_averaged_accuracies(
    data: Mapping[str, Mapping[str, Mapping[str, Mapping[str, List[List[float]]]]]],
    ylim: float = 1.0,
    metric_type: str = "nocot",  # "nocot" or "cot"
    aggregate_attributes: bool = True,  # True to average across attributes, False to plot separately
    fig_height: Optional[float] = None,  # Custom figure height
) -> None:
    """Create plots of accuracies averaged across attributes for each dataset.

    Args:
        data: The data dictionary from W&B
        ylim: Y-axis limit (default 1.0)
        metric_type: "nocot" or "cot"
        aggregate_attributes: If True, average across attributes. If False, plot each attribute separately
        fig_height: Custom figure height. If None, uses default (8 for aggregated, 16 for separated)
    """
    print(
        f"Creating averaged accuracy plots (metric_type: {metric_type}, aggregate_attributes: {aggregate_attributes})..."
    )

    # Set up the plot with appropriate height
    if fig_height is None:
        fig_height = (
            8 if aggregate_attributes else 16
        )  # Double height when showing attributes separately

    n_datasets = len(EXPERIMENT_CONFIG)
    fig, ax = plt.subplots(1, 1, figsize=(12, fig_height))

    # Define line styles for different attributes
    line_styles = [
        "-",
        "--",
        "-.",
        ":",
        (0, (3, 1, 1, 1)),
        (0, (5, 1)),
        (0, (3, 5, 1, 5)),
        (0, (1, 1)),
    ]

    # Process each dataset
    for dataset_idx, (config_path, config_info) in enumerate(EXPERIMENT_CONFIG.items()):
        label = config_info["label"]
        color = config_info["color"]
        attributes = config_info["attributes"]

        if aggregate_attributes:
            # Original behavior: collect and average all attributes
            acc_means_by_attr = []
            acc_sems_by_attr = []
            epochs_list = []

            for attr in attributes:
                # Accuracy data
                accuracy_key = "accuracy_cot" if metric_type == "cot" else "accuracy"
                if attr in data[config_path] and accuracy_key in data[config_path][attr]:
                    values = data[config_path][attr][accuracy_key][config_path]
                    epochs, means, sems = compute_statistics(values)
                    if len(epochs) > 0:
                        acc_means_by_attr.append(means)
                        acc_sems_by_attr.append(sems)
                        epochs_list.append(epochs)

            # Average across attributes if we have data
            if acc_means_by_attr:
                # Use the first epochs array (they should all be the same)
                epochs = epochs_list[0]

                # Average the means across attributes
                avg_acc_means = np.mean(acc_means_by_attr, axis=0)
                # Propagate error: sqrt(sum of squared sems) / sqrt(n_attributes)
                avg_acc_sems = np.sqrt(np.sum(np.square(acc_sems_by_attr), axis=0)) / np.sqrt(
                    len(acc_means_by_attr)
                )

                # Plot accuracy
                ax.plot(
                    epochs,
                    avg_acc_means,
                    label=f"{label}",
                    color=color,
                    linewidth=3,
                )
                ax.fill_between(
                    epochs,
                    avg_acc_means - avg_acc_sems,
                    avg_acc_means + avg_acc_sems,
                    color=color,
                    alpha=0.2,
                )
        else:
            # New behavior: plot each attribute separately
            for attr_idx, attr in enumerate(attributes):
                # Use cycling line styles if we run out
                line_style = line_styles[attr_idx % len(line_styles)]

                # Accuracy data
                accuracy_key = "accuracy_cot" if metric_type == "cot" else "accuracy"
                if attr in data[config_path] and accuracy_key in data[config_path][attr]:
                    values = data[config_path][attr][accuracy_key][config_path]
                    epochs, means, sems = compute_statistics(values)
                    if len(epochs) > 0:
                        # Label format: "Dataset - attribute" for first, "    attribute" for others
                        attr_label = f"{label} - {attr}" if attr_idx == 0 else f"    {attr}"
                        ax.plot(
                            epochs,
                            means,
                            label=attr_label,
                            color=color,
                            linestyle=line_style,
                            linewidth=2.5,
                        )
                        ax.fill_between(
                            epochs,
                            means - sems,
                            means + sems,
                            color=color,
                            alpha=0.1,
                        )

    # Customize plot
    ax.set_xlabel("Epoch", fontsize=22)
    ax.set_ylabel("Average Accuracy", fontsize=22)
    title_suffix = " (CoT)" if metric_type == "cot" else " (No-CoT)"
    if aggregate_attributes:
        ax.set_title(f"Average Accuracy{title_suffix}, by e2_type", fontsize=24, pad=20)
    else:
        ax.set_title(f"Accuracy{title_suffix}, by e2_type and e3_type", fontsize=24, pad=20)

    # Apply consistent styling
    apply_plot_style(ax, remove_grid=True, spine_width=2, tick_size=20)

    legend = ax.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12 if not aggregate_attributes else 16
    )
    # Add legend title
    legend_title = "e2_type" if aggregate_attributes else "e2_type - e3_type"
    legend.set_title(legend_title)
    legend.get_title().set_fontsize(14 if not aggregate_attributes else 18)
    legend.get_title().set_fontweight("bold")
    ax.set_ylim(0, ylim)

    plt.tight_layout()
    suffix = "_by_e2_type" if aggregate_attributes else "_by_e3_detailed_type"
    output_filename = f"semi_synthetic/{metric_type}_accuracy{suffix}.pdf"
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    plt.show()


def plot_by_meta_e3_categories(
    data: Mapping[str, Mapping[str, Mapping[str, Mapping[str, List[List[float]]]]]],
    plot_type: str = "accuracy",  # "accuracy", "loss_difference"
    metric_type: str = "nocot",  # "nocot" or "cot" (only for accuracy)
    ylim: Optional[float] = None,
    fig_height: float = 8,
) -> None:
    """Create plots, by meta-E3 categories (attribute types).

    Args:
        data: The data dictionary from W&B
        plot_type: "accuracy" or "loss_difference"
        metric_type: "nocot" or "cot" (only relevant for accuracy plots)
        ylim: Optional y-axis limit
        fig_height: Figure height
    """
    print(
        f"Creating plots by meta-E3 categories (plot_type: {plot_type}, metric_type: {metric_type})..."
    )

    # Collect data by meta-category
    meta_category_data = defaultdict(lambda: {"epochs": [], "means": [], "sems": [], "sources": []})

    # Process each dataset
    for config_path, config_info in EXPERIMENT_CONFIG.items():
        dataset_label = config_info["label"]
        attributes = config_info["attributes"]

        for attr in attributes:
            # Skip if attribute not in mapping
            if attr not in E3_TO_META:
                print(
                    f"Warning: Attribute '{attr}' from {dataset_label} not in meta-category mapping"
                )
                continue

            meta_category = E3_TO_META[attr]

            if plot_type == "accuracy":
                # Get accuracy data
                accuracy_key = "accuracy_cot" if metric_type == "cot" else "accuracy"
                if attr in data[config_path] and accuracy_key in data[config_path][attr]:
                    values = data[config_path][attr][accuracy_key][config_path]
                    epochs, means, sems = compute_statistics(values)
                    if len(epochs) > 0:
                        meta_category_data[meta_category]["epochs"].append(epochs)
                        meta_category_data[meta_category]["means"].append(means)
                        meta_category_data[meta_category]["sems"].append(sems)
                        meta_category_data[meta_category]["sources"].append(
                            f"{dataset_label}:{attr}"
                        )

            elif plot_type == "loss_difference":
                # Calculate loss difference
                if (
                    attr in data[config_path]
                    and "normal" in data[config_path][attr]
                    and "shuffled" in data[config_path][attr]
                ):
                    normal_values = data[config_path][attr]["normal"][config_path]
                    shuffled_values = data[config_path][attr]["shuffled"][config_path]

                    normal_epochs, normal_means, normal_sems = compute_statistics(normal_values)
                    shuffled_epochs, shuffled_means, shuffled_sems = compute_statistics(
                        shuffled_values
                    )

                    if len(normal_epochs) > 0 and len(shuffled_epochs) > 0:
                        # Calculate difference
                        diff_means = shuffled_means - normal_means
                        # Propagate error
                        diff_sems = np.sqrt(shuffled_sems**2 + normal_sems**2)

                        meta_category_data[meta_category]["epochs"].append(normal_epochs)
                        meta_category_data[meta_category]["means"].append(diff_means)
                        meta_category_data[meta_category]["sems"].append(diff_sems)
                        meta_category_data[meta_category]["sources"].append(
                            f"{dataset_label}:{attr}"
                        )

    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, fig_height))

    # Use tab20 colours and cycle through line styles for >20 categories
    base_colors = sns.color_palette("tab20", n_colors=20)
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

    # Plot each meta-category
    for idx, (meta_category, category_data) in enumerate(sorted(meta_category_data.items())):
        if not category_data["means"]:
            continue

        # Average across all instances of this meta-category
        # Assuming all have the same epochs (which should be true)
        epochs = category_data["epochs"][0]

        # Stack all means and sems
        all_means = np.array(category_data["means"])
        all_sems = np.array(category_data["sems"])

        # Calculate average
        avg_means = np.mean(all_means, axis=0)
        # Propagate error for averaging
        avg_sems = np.sqrt(np.sum(np.square(all_sems), axis=0)) / np.sqrt(len(all_means))

        # Create label with count of sources
        n_sources = len(category_data["sources"])
        label = f"{meta_category}"

        # Select colour and line style
        color_idx = idx % 20
        line_style_idx = idx // 20
        color = base_colors[color_idx]
        line_style = line_styles[line_style_idx % len(line_styles)]

        # Plot
        ax.plot(epochs, avg_means, label=label, color=color, linestyle=line_style, linewidth=4)
        ax.fill_between(epochs, avg_means - avg_sems, avg_means + avg_sems, color=color, alpha=0.2)

    # Customize plot
    ax.set_xlabel("Epoch", fontsize=22)

    if plot_type == "accuracy":
        ax.set_ylabel("Average Accuracy", fontsize=22)
        title_suffix = " (CoT)" if metric_type == "cot" else " (No-CoT)"
        ax.set_title(f"Average Accuracy{title_suffix}, by e3_type", fontsize=24, pad=20)
        if ylim is None:
            ax.set_ylim(0, 1.0)
    else:  # loss_difference
        ax.set_ylabel("Loss Advantage over Random Baseline", fontsize=22)
        ax.set_title("Loss Advantage over Random Baseline, by e3_type", fontsize=24, pad=20)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

    if ylim is not None:
        ax.set_ylim(0, ylim)

    # Apply consistent styling
    apply_plot_style(ax, remove_grid=True, spine_width=2, tick_size=20)

    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    # Add legend title
    legend_title = "e3_type"
    legend.set_title(legend_title)
    legend.get_title().set_fontsize(14)
    legend.get_title().set_fontweight("bold")

    plt.tight_layout()

    # Save with descriptive filename
    if plot_type == "accuracy":
        filename = f"semi_synthetic/{metric_type}_accuracy_by_e3_type.pdf"
    else:
        filename = "semi_synthetic/loss_advantage_by_e3_type.pdf"

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()


def plot_overall_accuracy_comparison(
    data: Mapping[str, Mapping[str, Mapping[str, Mapping[str, List[List[float]]]]]],
    epoch: Optional[int] = None,  # Which epoch to use, None for last epoch
) -> None:
    """Create a bar plot comparing first hop, second hop, and two-hop accuracies.

    Shows 4 bars:
    1. Accuracy A (first hop)
    2. Accuracy B (second hop from CSV)
    3. Two-hop No-CoT
    4. Two-hop CoT

    Args:
        data: The data dictionary from W&B
        epoch: Which epoch to use for comparison. If None, uses the last epoch.
    """
    print(
        f"Creating overall accuracy comparison bar plot (epoch: {epoch if epoch is not None else 'last'})..."
    )

    # Collect all accuracy values
    acc_a_values = []  # First hop
    nocot_values = []  # Two-hop No-CoT
    cot_values = []  # Two-hop CoT

    # Process each dataset for W&B data
    for config_path, config_info in EXPERIMENT_CONFIG.items():
        attributes = config_info["attributes"]

        for attr in attributes:
            # Get first hop accuracy (acc_a)
            if attr in data[config_path] and "accuracy_a" in data[config_path][attr]:
                values = data[config_path][attr]["accuracy_a"][config_path]
                epochs, means, sems = compute_statistics(values)
                if len(epochs) > 0:
                    # Use specified epoch or last epoch
                    if epoch is not None and epoch in epochs:
                        idx = np.where(epochs == epoch)[0][0]
                    else:
                        idx = -1  # Last epoch
                    acc_a_values.append(means[idx])

            # Get No-CoT accuracy
            if attr in data[config_path] and "accuracy" in data[config_path][attr]:
                values = data[config_path][attr]["accuracy"][config_path]
                epochs, means, sems = compute_statistics(values)
                if len(epochs) > 0:
                    # Use specified epoch or last epoch
                    if epoch is not None and epoch in epochs:
                        idx = np.where(epochs == epoch)[0][0]
                    else:
                        idx = -1  # Last epoch
                    nocot_values.append(means[idx])

            # Get CoT accuracy
            if attr in data[config_path] and "accuracy_cot" in data[config_path][attr]:
                values = data[config_path][attr]["accuracy_cot"][config_path]
                epochs, means, sems = compute_statistics(values)
                if len(epochs) > 0:
                    # Use specified epoch or last epoch
                    if epoch is not None and epoch in epochs:
                        idx = np.where(epochs == epoch)[0][0]
                    else:
                        idx = -1  # Last epoch
                    cot_values.append(means[idx])

    # Load second hop accuracy from CSV
    print("Loading second hop accuracy from CSV...")
    df = pd.read_csv("../second_hop_evaluation_results.csv")
    # Calculate accuracy by e2_type and e3_type
    acc_b_by_dataset_attr = df.groupby(["e2_type", "e3_type"])["correct"].mean()
    acc_b_values = acc_b_by_dataset_attr.values.tolist()

    # Calculate overall means and standard errors
    acc_a_mean = np.mean(acc_a_values) if acc_a_values else 0
    acc_a_sem = np.std(acc_a_values) / np.sqrt(len(acc_a_values)) if acc_a_values else 0

    acc_b_mean = np.mean(acc_b_values) if acc_b_values else 0
    acc_b_sem = np.std(acc_b_values) / np.sqrt(len(acc_b_values)) if acc_b_values else 0

    nocot_mean = np.mean(nocot_values) if nocot_values else 0
    nocot_sem = np.std(nocot_values) / np.sqrt(len(nocot_values)) if nocot_values else 0

    cot_mean = np.mean(cot_values) if cot_values else 0
    cot_sem = np.std(cot_values) / np.sqrt(len(cot_values)) if cot_values else 0

    # Create bar plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # Bar positions - adjusted for better spacing
    x = np.array([0, 1, 2, 3])
    width = 0.75  # Wider bars like in the paper

    # Adjust spacing between bars

    means = np.array([acc_a_mean, acc_b_mean, cot_mean, nocot_mean])
    sems = np.array([acc_a_sem, acc_b_sem, cot_sem, nocot_sem])
    labels = [
        "1st hop\n(synthetic)",
        "2nd hop\n(real-world)",
        "Two-Hop\nwith CoT",
        "Two-Hop\nwithout CoT",
    ]

    # Use different shades of blue
    base_color = "#1f77b4"  # Base blue color
    colors = [
        "#B8D4F1",  # Pale blue for 1st hop
        "#B8D4F1",  # Pale blue for 2nd hop
        "#B8D4F1",  # Pale blue for Two-Hop with CoT
        "#084594",  # Dark blue for Two-Hop without CoT (keep vibrant)
    ]

    # Create bars
    bars = ax.bar(
        x,
        means,
        width,
        yerr=sems,
        capsize=5,  # Smaller capsize like in paper
        color=colors,
        error_kw={"linewidth": 2, "ecolor": "black"},
    )

    # # Add value labels on bars
    # for i, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
    #     height = bar.get_height()
    #     ax.text(
    #         bar.get_x() + bar.get_width() / 2.0,
    #         height + sem + 0.01,
    #         f"{mean:.3f}",
    #         ha="center",
    #         va="bottom",
    #         fontsize=16,
    #     )

    # Customize plot
    ax.set_ylabel("Average Accuracy", fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_ylim(0, 1.0)

    # Apply consistent styling
    apply_plot_style(ax, remove_grid=True, spine_width=2, tick_size=20)

    # Use seaborn despine for cleaner look
    sns.despine()

    plt.tight_layout()
    plt.savefig("semi_synthetic_avg_acc.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    # Print statistics
    print(f"\nStatistics:")
    print(f"Accuracy A (First Hop): {acc_a_mean:.4f} ± {acc_a_sem:.4f} (n={len(acc_a_values)})")
    print(f"Accuracy B (Second Hop): {acc_b_mean:.4f} ± {acc_b_sem:.4f} (n={len(acc_b_values)})")
    print(f"Two-Hop No-CoT: {nocot_mean:.4f} ± {nocot_sem:.4f} (n={len(nocot_values)})")
    print(f"Two-Hop CoT: {cot_mean:.4f} ± {cot_sem:.4f} (n={len(cot_values)})")
    print(
        f"Improvement (No-CoT to CoT): {cot_mean - nocot_mean:.4f} ({(cot_mean - nocot_mean) / nocot_mean * 100:.1f}%)"
    )


def plot_overall_loss_comparison(
    data: Mapping[str, Mapping[str, Mapping[str, Mapping[str, List[List[float]]]]]],
    ylim: Optional[float] = None,
) -> None:
    """Create a line plot comparing overall shuffled vs non-shuffled loss averaged across all datasets and attributes.

    Args:
        data: The data dictionary from W&B
        ylim: Optional y-axis limit
    """
    print("Creating overall loss comparison plot (shuffled vs non-shuffled)...")

    # Collect all loss values by epoch
    normal_means_by_epoch = defaultdict(list)
    shuffled_means_by_epoch = defaultdict(list)

    # Process each dataset
    for config_path, config_info in EXPERIMENT_CONFIG.items():
        attributes = config_info["attributes"]

        for attr in attributes:
            # Get normal loss
            if attr in data[config_path] and "normal" in data[config_path][attr]:
                values = data[config_path][attr]["normal"][config_path]
                epochs, means, sems = compute_statistics(values)
                if len(epochs) > 0:
                    for epoch, mean in zip(epochs, means):
                        normal_means_by_epoch[epoch].append(mean)

            # Get shuffled loss
            if attr in data[config_path] and "shuffled" in data[config_path][attr]:
                values = data[config_path][attr]["shuffled"][config_path]
                epochs, means, sems = compute_statistics(values)
                if len(epochs) > 0:
                    for epoch, mean in zip(epochs, means):
                        shuffled_means_by_epoch[epoch].append(mean)

    # Calculate overall means and SEMs for each epoch
    epochs = sorted(normal_means_by_epoch.keys())
    normal_means = []
    normal_sems = []
    shuffled_means = []
    shuffled_sems = []

    for epoch in epochs:
        # Normal loss
        if epoch in normal_means_by_epoch:
            values = normal_means_by_epoch[epoch]
            normal_means.append(np.mean(values))
            normal_sems.append(np.std(values) / np.sqrt(len(values)))

        # Shuffled loss
        if epoch in shuffled_means_by_epoch:
            values = shuffled_means_by_epoch[epoch]
            shuffled_means.append(np.mean(values))
            shuffled_sems.append(np.std(values) / np.sqrt(len(values)))

    # Convert to numpy arrays
    epochs = np.array(epochs)
    normal_means = np.array(normal_means)
    normal_sems = np.array(normal_sems)
    shuffled_means = np.array(shuffled_means)
    shuffled_sems = np.array(shuffled_sems)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot normal loss (solid line)
    ax.plot(epochs, normal_means, label="Non-shuffled", color="#084594", linewidth=4)
    ax.fill_between(
        epochs, normal_means - normal_sems, normal_means + normal_sems, color="#084594", alpha=0.2
    )

    # Plot shuffled loss (dotted line)
    ax.plot(epochs, shuffled_means, label="Shuffled", color="#084594", linestyle=":", linewidth=4)
    ax.fill_between(
        epochs,
        shuffled_means - shuffled_sems,
        shuffled_means + shuffled_sems,
        color="#084594",
        alpha=0.2,
    )

    # Customize plot
    ax.set_xlabel("Epoch", fontsize=22)
    ax.set_ylabel("Average Loss", fontsize=22)
    # ax.set_title("Overall Loss Comparison: Shuffled vs Non-shuffled", fontsize=24, pad=20)

    # Apply consistent styling
    apply_plot_style(ax, remove_grid=True, spine_width=2, tick_size=20)

    # Add legend
    # legend = ax.legend(fontsize=18, loc="upper right")
    # legend.get_frame().set_linewidth(1.5)

    if ylim:
        ax.set_ylim(0, ylim)

    plt.tight_layout()
    plt.savefig("semi_synthetic_avg_loss.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    # Print statistics for final epoch
    print(f"\nFinal epoch statistics:")
    print(f"Non-shuffled: {normal_means[-1]:.4f} ± {normal_sems[-1]:.4f}")
    print(f"Shuffled: {shuffled_means[-1]:.4f} ± {shuffled_sems[-1]:.4f}")
    print(f"Difference: {shuffled_means[-1] - normal_means[-1]:.4f}")


if __name__ == "__main__":
    print("Entering __main__")

    # %%
    print("Fetching data from W&B...")
    data = get_metrics_for_runs()

    # %%

    os.makedirs("semi_synthetic", exist_ok=True)

    # %%
    # Plot losses with auto scaling
    plot_losses(data)

    # %%
    # Plot accuracies (no-CoT)
    plot_accuracies(data)

    # %%
    # Plot accuracies (CoT)
    plot_accuracies(data, metric_type="cot")

    # %%
    # Plot averaged losses across attributes
    plot_averaged_losses(data)

    # # %%
    # # Plot averaged loss advantage over random baseline (shuffled - non-shuffled)
    plot_averaged_losses(data, mode="difference")

    # # %%
    # # Plot averaged accuracies across attributes (no-CoT)
    plot_averaged_accuracies(data, metric_type="nocot")

    # # %%
    # # Plot averaged accuracies across attributes (CoT)
    plot_averaged_accuracies(data, metric_type="cot")

    # %%
    # Plot loss advantage with attributes shown separately
    plot_averaged_losses(data, mode="difference", aggregate_attributes=False)

    # %%
    # Plot accuracies (no-CoT) with attributes shown separately
    plot_averaged_accuracies(data, metric_type="nocot", aggregate_attributes=False)

    # %%
    # Plot accuracies (CoT) with attributes shown separately
    plot_averaged_accuracies(data, metric_type="cot", aggregate_attributes=False)

    # %%
    # # Plot by meta-E3 categories - Loss Advantage over Random Baseline
    plot_by_meta_e3_categories(data, plot_type="loss_difference")

    # %%
    # # Plot by meta-E3 categories - No-CoT Accuracy
    plot_by_meta_e3_categories(data, plot_type="accuracy", metric_type="nocot")

    # %%
    # # Plot by meta-E3 categories - CoT Accuracy
    plot_by_meta_e3_categories(data, plot_type="accuracy", metric_type="cot")

    # %%
    # Plot overall accuracy comparison
    plot_overall_accuracy_comparison(data)

    # %%
    # Plot overall loss comparison
    plot_overall_loss_comparison(data)

# %%
