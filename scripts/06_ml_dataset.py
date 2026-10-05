"""
Step 06: the ML-ready dataset used by notebooks/02_ML_baseline.ipynb.

Inputs (from 04_features.py):
  features/<cohort>_features.tsv    methylation fraction per sample x region
  features/<cohort>_coverage.tsv    methylation calls per sample x region
  features/<cohort>_n_cpg.tsv       observed CpG sites per sample x region
  metadata/<cohort>.tsv             clinical / SRA metadata

Outputs (notebooks/data/):
  X_methylation.tsv   sample_id x kept regions, low-depth values set to NaN
  X_coverage.tsv      same shape, methylation calls (for weighting / QC)
  samples.tsv         labels + covariates, one row per sample in X
  regions.tsv         all 2473 regions, with QC stats and kept/drop reason
  dataset_info.json   parameters and counts

In nfcore mode (CTDNA_MODE=nfcore) the inputs are read from features/nfcore/
and the outputs go to notebooks/data/nfcore/.

Deliberately NOT done here (must happen inside cross-validation folds to
avoid information leakage): imputation, scaling, feature selection.
Only label-free filters are applied (per-value depth mask, region
missingness, constant regions).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from utils import ROOT, MODE, COHORT, COHORT_FILE, TARGET_BED, FEATURES_DIR, ML_DATA_DIR


FEATURE_FILE = FEATURES_DIR / f"{COHORT}_features.tsv"
COVERAGE_FILE = FEATURES_DIR / f"{COHORT}_coverage.tsv"
N_CPG_FILE = FEATURES_DIR / f"{COHORT}_n_cpg.tsv"

META_COLUMNS = ["run_id", "sample_id", "ml_label"]

# Metadata carried into samples.tsv. These are for stratification,
# confounder checks and optional covariates -- not methylation features.
#
# Intentionally excluded:
#   predicted     output of the published ELSA classifier -> label leakage
#   ground_truth  same information as ml_label
SAMPLE_COLUMNS = [
    "sample_id",
    "run_id",
    "ml_label",
    "disease_status",
    "stage",
    "cancer_type",
    "model_group",
    "age",
    "gender",
    "dna_input_ng",
    "dna_concentration_ng_ul",
]


def parse_args():

    p = argparse.ArgumentParser(
        description="Build the ML-ready dataset in notebooks/data/ "
                    "(notebooks/data/nfcore/ in nfcore mode)."
    )

    p.add_argument(
        "--min-depth",
        type=int,
        default=20,
        help="Set a region value to NaN if it has fewer methylation calls "
             "than this (default: 20).",
    )
    p.add_argument(
        "--max-missing",
        type=float,
        default=0.2,
        help="Drop regions missing in more than this fraction of samples "
             "after depth masking (default: 0.2).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ML_DATA_DIR,
        help="Output directory (default: notebooks/data, "
             "or notebooks/data/nfcore in nfcore mode).",
    )

    return p.parse_args()


def load_matrix(path):

    df = pd.read_csv(path, sep="\t")

    meta = df[META_COLUMNS].copy()
    values = df.drop(columns=META_COLUMNS).set_index(df["sample_id"])
    values.index.name = "sample_id"

    return meta, values


def main():

    args = parse_args()

    out_dir = ROOT / args.out_dir   # absolute paths override ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PREPARE ML DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    meta, fraction = load_matrix(FEATURE_FILE)
    _, coverage = load_matrix(COVERAGE_FILE)
    _, n_cpg = load_matrix(N_CPG_FILE)

    if not (
        fraction.index.equals(coverage.index)
        and fraction.columns.equals(coverage.columns)
        and fraction.index.equals(n_cpg.index)
    ):
        raise ValueError(
            "Feature / coverage / n_cpg matrices are not aligned. "
            "Rerun 04_features.py."
        )

    region_ids = fraction.columns.tolist()

    print(f"Samples:  {len(fraction)}")
    print(f"Regions:  {len(region_ids)}")

    # --------------------------------------------------------
    # Sample metadata
    # --------------------------------------------------------

    cohort = pd.read_csv(COHORT_FILE, sep="\t")

    samples = (
        meta[["sample_id", "run_id"]]
        .merge(
            cohort[[c for c in SAMPLE_COLUMNS if c in cohort.columns]],
            on=["sample_id", "run_id"],
            how="left",
            validate="one_to_one",
        )
    )

    if not (samples["ml_label"].values == meta["ml_label"].values).all():
        raise ValueError("ml_label in feature matrix and cohort disagree.")

    # SRA submission series (SRR998xxxx vs SRR151xxxx). Same BioProject
    # and instrument, but label balance differs between series, so it is
    # kept as a potential batch / confounder variable.
    samples["submission_series"] = samples["run_id"].str[:6]

    samples["median_region_depth"] = coverage.median(axis=1).values
    samples["mean_region_methylation"] = fraction.mean(axis=1).values

    # --------------------------------------------------------
    # Depth masking (per cell, label-free)
    # --------------------------------------------------------

    low_depth = coverage < args.min_depth
    X = fraction.mask(low_depth)

    masked = int((low_depth & fraction.notna()).sum().sum())

    print(
        f"\nDepth mask (< {args.min_depth} calls): "
        f"{masked} additional values set to NaN"
    )

    # --------------------------------------------------------
    # Region filters (label-free)
    # --------------------------------------------------------

    targets = pd.read_csv(
        TARGET_BED,
        sep="\t",
        header=None,
        names=["chrom", "start0", "end", "region_id"],
    )

    regions = targets.set_index("region_id").loc[region_ids].reset_index()
    regions["length"] = regions["end"] - regions["start0"]
    regions["median_n_cpg"] = n_cpg.median(axis=0).values
    regions["median_depth"] = coverage.median(axis=0).values
    regions["missing_fraction"] = X.isna().mean(axis=0).values
    regions["variance"] = X.var(axis=0, skipna=True).values
    regions["unique_values"] = X.nunique(axis=0, dropna=True).values

    drop_reason = np.where(
        regions["missing_fraction"] > args.max_missing,
        f"missing>{args.max_missing}",
        np.where(
            regions["unique_values"] <= 1,
            "constant",
            "",
        ),
    )

    regions["drop_reason"] = drop_reason
    regions["kept"] = regions["drop_reason"] == ""

    kept = regions.loc[regions["kept"], "region_id"].tolist()

    print(f"Regions dropped (missing > {args.max_missing:.0%}): "
          f"{(regions['drop_reason'].str.startswith('missing')).sum()}")
    print(f"Regions dropped (constant): "
          f"{(regions['drop_reason'] == 'constant').sum()}")
    print(f"Regions kept: {len(kept)}")

    X = X[kept]
    X_coverage = coverage[kept]

    remaining_missing = float(X.isna().mean().mean())

    print(f"Remaining missing values: {remaining_missing * 100:.3f}% "
          f"(impute inside CV)")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    X.to_csv(out_dir / "X_methylation.tsv", sep="\t")
    X_coverage.to_csv(out_dir / "X_coverage.tsv", sep="\t")
    samples.to_csv(out_dir / "samples.tsv", sep="\t", index=False)
    regions.to_csv(out_dir / "regions.tsv", sep="\t", index=False)

    label_counts = samples["ml_label"].value_counts().sort_index()

    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cohort": COHORT,
        "mode": MODE,
        "source_files": [
            str(p.relative_to(ROOT))
            for p in (FEATURE_FILE, COVERAGE_FILE, N_CPG_FILE, COHORT_FILE)
        ],
        "params": {
            "min_depth": args.min_depth,
            "max_missing": args.max_missing,
        },
        "n_samples": int(len(X)),
        "n_regions_total": int(len(region_ids)),
        "n_regions_kept": int(len(kept)),
        "labels": {str(k): int(v) for k, v in label_counts.items()},
        "label_meaning": {"0": "non-cancer control", "1": "Stage I lung cancer"},
        "remaining_missing_fraction": remaining_missing,
        "notes": [
            "Values are coverage-weighted methylation fractions in [0, 1].",
            "Impute / scale / select features inside CV folds only.",
            "'predicted' (published classifier output) is intentionally excluded.",
        ],
    }

    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))

    print()
    print(f"Labels: " + ", ".join(f"{k}={v}" for k, v in label_counts.items()))
    print(f"Saved to: {out_dir}")
    for name in (
        "X_methylation.tsv", "X_coverage.tsv", "samples.tsv",
        "regions.tsv", "dataset_info.json",
    ):
        print(f"  {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
