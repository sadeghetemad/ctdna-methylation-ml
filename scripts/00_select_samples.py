from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "metadata/primary_cohort_stage1_vs_control.tsv"
OUTPUT = ROOT / "metadata/test_cohort_10.tsv"
ACCESSIONS = ROOT / "metadata/test_cohort_10_SRR.txt"

df = pd.read_csv(INPUT, sep="\t")

cancer = (
    df[df["ml_label"] == 1]
    .sample(n=5, random_state=42)
)

control = (
    df[df["ml_label"] == 0]
    .sample(n=5, random_state=42)
)

selected = pd.concat(
    [cancer, control],
    ignore_index=True
)

# Shuffle but keep reproducible
selected = selected.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

selected.to_csv(
    OUTPUT,
    sep="\t",
    index=False
)

selected["run_id"].to_csv(
    ACCESSIONS,
    index=False,
    header=False
)

print(
    selected[
        ["sample_id", "run_id", "disease_status", "stage", "ml_label"]
    ].to_string(index=False)
)

print("\nCounts:")
print(selected["ml_label"].value_counts().sort_index())

print(f"\nSaved: {OUTPUT}")
print(f"Saved: {ACCESSIONS}")
