# %%

from typing import Any

from inspect_ai.dataset import Sample, csv_dataset

from latent_reasoning.evaluate_api_models import (
    PATH_TO_HOP_TOO_LATE,
    SEED,
    create_sample,
    get_wh_word,
    n,
)


def record_to_sample_2hop(record: dict[str, Any]) -> Sample:
    wh_word = get_wh_word(record["e3_type"])
    return create_sample(
        f"{wh_word} is {record['source_prompt'].removesuffix(' is')}?",
        record["e3_aliases"],
        record["id"],
        record,
    )


dataset_2hop = csv_dataset(
    str(PATH_TO_HOP_TOO_LATE / "two_hop_filtered.csv"),
    sample_fields=record_to_sample_2hop,
    shuffle=True,
    seed=SEED,
)[:n]

# %%

from collections import Counter

# Entity type distributions
e1_types = Counter(sample.metadata["e1_type"] for sample in dataset_2hop)
e2_types = Counter(sample.metadata["e2_type"] for sample in dataset_2hop)
e3_types = Counter(sample.metadata["e3_type"] for sample in dataset_2hop)

# Question type distribution (combination of r1 and r2 templates)
question_types = Counter(
    (sample.metadata["r1_template"], sample.metadata["r2_template"]) for sample in dataset_2hop
)

print("\nEntity type distributions:")
print("\nE1 types:", dict(e1_types.most_common()))
print("\nE2 types:", dict(e2_types.most_common()))
print("\nE3 types:", dict(e3_types.most_common()))

most_common_question_types = question_types.most_common(20)
total_questions_covered = sum(count for _, count in most_common_question_types)

print(
    f"\nTop 20 question types (r1_template + r2_template; covering a total of {total_questions_covered} of {len(dataset_2hop)} questions):"
)
for (r1, r2), count in most_common_question_types:
    print(f"\nCount: {count}")
    print(f"R1: {r1}")
    print(f"R2: {r2}")
    r1_stripped = r1.replace("{}", "").rstrip()
    r2_stripped = r2.replace("{}", "").rstrip()
    print(f"Example: Who/What is {r2_stripped} {r1_stripped} X?")


# %%

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
# plt.style.use('seaborn')
sns.set_palette("muted")
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["font.size"] = 10


def plot_entity_distributions():
    # Get top 10 for each position
    e1_types = Counter(sample.metadata["e1_type"] for sample in dataset_2hop)
    e2_types = Counter(sample.metadata["e2_type"] for sample in dataset_2hop)
    e3_types = Counter(sample.metadata["e3_type"] for sample in dataset_2hop)

    # Create DataFrame
    df = pd.DataFrame(
        {"E1": pd.Series(e1_types), "E2": pd.Series(e2_types), "E3": pd.Series(e3_types)}
    ).fillna(0)

    # Set color palette
    colors = sns.color_palette("muted")[:3]  # Get first 3 colors for E1, E2, E3

    # Sort by total frequency
    df["total"] = df.sum(axis=1)
    df = df.sort_values("total", ascending=False).head(20)
    df = df.drop("total", axis=1)

    # Plot
    ax = df.plot(kind="bar", width=0.8, color=colors)
    plt.title("Distribution of Entity Types by Position")
    plt.xlabel("Entity Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Position")
    plt.tight_layout()
    sns.despine()

    plt.savefig("../entity_type_distribution.pdf")
    plt.show()


def plot_common_question_types():
    # Get question types
    question_types = Counter(
        (sample.metadata["r1_template"], sample.metadata["r2_template"]) for sample in dataset_2hop
    )

    # Create readable labels
    def format_question(r1, r2):
        r1_stripped = r1.replace("{}", "").rstrip()
        r2_stripped = r2.replace("{}", "").rstrip()
        return f"{r2_stripped} {r1_stripped} X"

    # Get top 10
    top_20 = question_types.most_common(20)[::-1]
    labels = [format_question(r1, r2) for (r1, r2), _ in top_20]
    counts = [count for _, count in top_20]

    color = "#808080"  # Standard gray

    # Plot
    plt.figure(figsize=[12, 6])
    y_pos = range(len(labels))
    bars = plt.barh(y_pos, counts, color=color)

    plt.yticks(y_pos, labels)
    plt.xlabel("Count")
    plt.title("Top 20 Most Common Question Types")
    plt.tight_layout()
    sns.despine()
    plt.savefig("../question_type_distribution.pdf")
    plt.show()


# Generate plots
plot_entity_distributions()
plot_common_question_types()

# %%
