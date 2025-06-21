# %%

import pandas as pd

df = pd.read_csv("two_hop_results_raw_hopping_too_late.csv")

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
