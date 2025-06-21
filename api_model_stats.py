# %%

from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_api_model_stats(keep_shortcuts: bool = False):
    models = [
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "claude-3-opus-20240229",
        "gpt-4o-2024-05-13",
    ]

    dfs = []
    for model in models:
        df = pd.read_csv("two_hop_results_raw_hopping_too_late_pivot.csv")

        df = df[df["model"] == model]
        # Keep only triplets where (1) both one-hops are correct and (2) the model does not use reasoning shortcuts
        df = df[df["both_1hops_correct"] == 1]
        if keep_shortcuts:
            df["no_cot_via_shortcuts"] = df["two_hop_no_cot_baseline1"].astype(int) | df[
                "two_hop_no_cot_baseline2"
            ].astype(int)
        else:
            df = df[df["two_hop_no_cot_baseline1"] == 0]
            df = df[df["two_hop_no_cot_baseline2"] == 0]

        df = df[df["e2_type"] != "country"]

        
        print(df.columns)

        group_key = ["r1_template", "r2_template"]
        # group_key = ["e2_type"]


        # Remove non-numeric columns, but KEEP ones we want to group by:
        df_pre_group = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_keep = list(numeric_cols) + group_key
        df_pre_group = df_pre_group[cols_to_keep]
        performance = (
            df_pre_group.groupby(group_key)
            .agg(
                {
                    **{col: "mean" for col in df_pre_group.columns if col not in group_key},
                    "two_hop_cot": [
                        "mean",
                        "size",
                    ],  # Using two_hop_cot as reference column for count
                }
            )
            .rename(columns={"size": "count"})
        )
        performance.columns = [
            col[0] if col[1] == "mean" else col[1] for col in performance.columns
        ]

        # Sort performance DataFrame by two_hop_no_cot values in descending order
        two_hop_no_cot = performance.sort_values(by="two_hop_no_cot", ascending=False)
        columns_to_show = [
            "two_hop_no_cot",
            "two_hop_cot",
            "count",
        ]
        if keep_shortcuts:
            columns_to_show.append("no_cot_via_shortcuts")
        two_hop_no_cot_values = two_hop_no_cot[columns_to_show]

        two_hop_no_cot_values["type"] = two_hop_no_cot_values.index.map(
            lambda x: f"{x[1].rstrip('}').rstrip('{').strip()} → {x[0].rstrip('}').rstrip('{').strip()}"
        )
        # two_hop_no_cot_values["type"] = two_hop_no_cot_values.index.map(
        #     lambda x: x
        # )

        # Reorder columns to make type first
        column_order = [
                "type",
                "two_hop_no_cot",
                "two_hop_cot",
        ]
        if keep_shortcuts:
            column_order.append("no_cot_via_shortcuts")
        column_order.append("count")
        two_hop_no_cot_values = two_hop_no_cot_values[column_order]

        # turn accuracies into percentages
        acc_fields = list(set(column_order) - {"count", "type"})
        two_hop_no_cot_values[acc_fields] *= 100

        # only keep rows where no-CoT acc is 0:
        # two_hop_no_cot_values = two_hop_no_cot_values[two_hop_no_cot_values["two_hop_no_cot"] <= 20]
        # two_hop_no_cot_values = two_hop_no_cot_values[two_hop_no_cot_values["two_hop_cot"] >= 20]

        model_name = {
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "llama405b",
            "claude-3-opus-20240229": "claude3opus",
            "gpt-4o-2024-05-13": "gpt4o",
        }[model]

        two_hop_no_cot_values = two_hop_no_cot_values.rename(
            columns={
                "two_hop_no_cot": f"{model_name}_no_cot",
                "two_hop_cot": f"{model_name}_cot",
                "count": f"{model_name}_count",
                "no_cot_via_shortcuts": f"{model_name}_no_cot_via_shortcuts",
            }
        )
        dfs.append(two_hop_no_cot_values)

    # use outer merge on "type"
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="type", how="outer"), dfs)
    merged_df = merged_df.sort_values(by="llama405b_no_cot", ascending=False)

    with pd.option_context("display.max_rows", None):
        print(merged_df.to_string(index=False))

    # plot table with all columns as a heatmap with colors. use type as row index:
    fig, ax = plt.subplots(figsize=(20, 40))
    # fig, ax = plt.subplots(figsize=(20, 20))

    sns.heatmap(
        merged_df.iloc[:, 1:],
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        ax=ax,
        cbar=False,
        yticklabels=merged_df["type"],
    )
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.xaxis.set_ticklabels(ax.get_xticklabels(), rotation=45, ha="left", va="bottom")

    ax.set_yticklabels(merged_df["type"], rotation=0)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Type of e2")
    return fig, ax


get_api_model_stats(keep_shortcuts=False)
