"""
Step 05: QC tables of the feature matrix written by step 04.
"""

import numpy as np
import pandas as pd

from utils import COHORT, EXPECTED_REGIONS, FEATURES_DIR, FEATURE_QC_DIR


# ============================================================
# Paths
# ============================================================

FEATURE_FILE = FEATURES_DIR / f"{COHORT}_features.tsv"

# Reads per region, also written by step 04; optional.
COVERAGE_FILE = FEATURES_DIR / f"{COHORT}_coverage.tsv"

OUT_DIR = FEATURE_QC_DIR    # results/feature_qc/ (custom) or .../nfcore/
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("FEATURE MATRIX QC")
    print("=" * 70)

    # --------------------------------------------------------
    # Load feature matrix
    # --------------------------------------------------------

    df = pd.read_csv(
        FEATURE_FILE,
        sep="\t",
    )

    feature_cols = [
        col
        for col in df.columns
        if col.startswith("ELSA_")
    ]

    if not feature_cols:
        raise ValueError(
            "No ELSA feature columns found."
        )

    X = df[feature_cols]

    print(f"Samples:  {len(df)}")
    print(f"Features: {len(feature_cols)}")

    if len(feature_cols) != EXPECTED_REGIONS:
        raise ValueError(
            f"Expected {EXPECTED_REGIONS} ELSA features, "
            f"found {len(feature_cols)}"
        )

    # --------------------------------------------------------
    # Basic value validation
    # --------------------------------------------------------

    observed = X.stack().dropna()   # pandas >= 3 keeps NaN in stack()

    if observed.empty:
        raise RuntimeError(
            "No observed methylation values found."
        )

    invalid = (
        (observed < 0)
        | (observed > 1)
    )

    invalid_count = int(
        invalid.sum()
    )

    print()
    print("VALUE RANGE")
    print("-" * 70)

    print(
        f"Minimum methylation: "
        f"{observed.min():.6f}"
    )

    print(
        f"Maximum methylation: "
        f"{observed.max():.6f}"
    )

    print(
        f"Values outside [0,1]: "
        f"{invalid_count}"
    )

    if invalid_count > 0:
        raise RuntimeError(
            "Methylation values outside [0,1] detected."
        )

    # --------------------------------------------------------
    # Overall missingness
    # --------------------------------------------------------

    total_cells = X.size

    missing_cells = int(
        X.isna().sum().sum()
    )

    missing_percent = (
        missing_cells
        / total_cells
        * 100
    )

    print()
    print("OVERALL MISSINGNESS")
    print("-" * 70)

    print(
        f"Missing: "
        f"{missing_cells}/{total_cells} "
        f"({missing_percent:.2f}%)"
    )

    # --------------------------------------------------------
    # Missingness by sample
    # --------------------------------------------------------

    sample_qc = pd.DataFrame({
        "run_id": df["run_id"],
    })

    if "sample_id" in df.columns:
        sample_qc["sample_id"] = (
            df["sample_id"]
        )

    if "ml_label" in df.columns:
        sample_qc["ml_label"] = (
            df["ml_label"]
        )

    sample_qc["missing_features"] = (
        X.isna()
        .sum(axis=1)
        .values
    )

    sample_qc["observed_features"] = (
        X.notna()
        .sum(axis=1)
        .values
    )

    sample_qc["missing_percent"] = (
        sample_qc["missing_features"]
        / len(feature_cols)
        * 100
    )

    if COVERAGE_FILE.exists():

        coverage = (
            pd.read_csv(COVERAGE_FILE, sep="\t")
            .set_index("run_id")
            .loc[df["run_id"], feature_cols]
        )

        sample_qc["median_region_depth"] = (
            coverage.median(axis=1).values
        )

        sample_qc["regions_depth_ge_20"] = (
            (coverage >= 20).sum(axis=1).values
        )

    print()
    print("MISSINGNESS BY SAMPLE")
    print("-" * 70)

    print(
        sample_qc.to_string(
            index=False
        )
    )

    sample_qc.to_csv(
        OUT_DIR / "missingness_by_sample.tsv",
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Missingness by region
    # --------------------------------------------------------

    region_qc = pd.DataFrame({
        "region_id": feature_cols,
        "missing_samples": (
            X.isna()
            .sum(axis=0)
            .values
        ),
        "observed_samples": (
            X.notna()
            .sum(axis=0)
            .values
        ),
    })

    region_qc["missing_percent"] = (
        region_qc["missing_samples"]
        / len(df)
        * 100
    )

    region_qc = region_qc.sort_values(
        [
            "missing_samples",
            "region_id",
        ],
        ascending=[
            False,
            True,
        ],
    )

    region_qc.to_csv(
        OUT_DIR / "missingness_by_region.tsv",
        sep="\t",
        index=False,
    )

    regions_with_missing = int(
        (
            region_qc["missing_samples"] > 0
        ).sum()
    )

    completely_missing_regions = int(
        (
            region_qc["missing_samples"]
            == len(df)
        ).sum()
    )

    print()
    print("REGION MISSINGNESS")
    print("-" * 70)

    print(
        f"Regions with >=1 missing sample: "
        f"{regions_with_missing}"
    )

    print(
        f"Regions missing in ALL samples: "
        f"{completely_missing_regions}"
    )

    print()
    print("Top regions by missingness:")

    print(
        region_qc.head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Feature variance
    # --------------------------------------------------------

    variances = X.var(
        axis=0,
        skipna=True,
    )

    unique_counts = X.nunique(
        axis=0,
        dropna=True,
    )

    variance_qc = pd.DataFrame({
        "region_id": feature_cols,
        "variance": variances.values,
        "unique_values": unique_counts.values,
    })

    zero_variance_mask = (
        variance_qc["unique_values"] <= 1
    )

    zero_variance_count = int(
        zero_variance_mask.sum()
    )

    variance_qc["zero_variance"] = (
        zero_variance_mask
    )

    variance_qc.to_csv(
        OUT_DIR / "feature_variance.tsv",
        sep="\t",
        index=False,
    )

    print()
    print("FEATURE VARIANCE")
    print("-" * 70)

    print(
        f"Zero-variance / constant regions: "
        f"{zero_variance_count}"
    )

    # --------------------------------------------------------
    # Per-feature methylation summary
    # --------------------------------------------------------

    feature_summary = pd.DataFrame({
        "region_id": feature_cols,
        "mean": X.mean(
            axis=0,
            skipna=True,
        ).values,
        "median": X.median(
            axis=0,
            skipna=True,
        ).values,
        "std": X.std(
            axis=0,
            skipna=True,
        ).values,
        "min": X.min(
            axis=0,
            skipna=True,
        ).values,
        "max": X.max(
            axis=0,
            skipna=True,
        ).values,
    })

    feature_summary.to_csv(
        OUT_DIR / "feature_summary.tsv",
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Sample-level methylation distribution
    # --------------------------------------------------------

    sample_distribution = pd.DataFrame({
        "run_id": df["run_id"],
    })

    if "sample_id" in df.columns:
        sample_distribution["sample_id"] = (
            df["sample_id"]
        )

    if "ml_label" in df.columns:
        sample_distribution["ml_label"] = (
            df["ml_label"]
        )

    sample_distribution["mean_methylation"] = (
        X.mean(
            axis=1,
            skipna=True,
        ).values
    )

    sample_distribution["median_methylation"] = (
        X.median(
            axis=1,
            skipna=True,
        ).values
    )

    sample_distribution["std_methylation"] = (
        X.std(
            axis=1,
            skipna=True,
        ).values
    )

    sample_distribution["min_methylation"] = (
        X.min(
            axis=1,
            skipna=True,
        ).values
    )

    sample_distribution["max_methylation"] = (
        X.max(
            axis=1,
            skipna=True,
        ).values
    )

    sample_distribution.to_csv(
        OUT_DIR / "sample_methylation_summary.tsv",
        sep="\t",
        index=False,
    )

    print()
    print("SAMPLE METHYLATION SUMMARY")
    print("-" * 70)

    print(
        sample_distribution.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Missingness by ML class
    # --------------------------------------------------------

    if "ml_label" in df.columns:

        print()
        print("MISSINGNESS BY ML LABEL")
        print("-" * 70)

        class_rows = []

        for label in sorted(
            df["ml_label"].dropna().unique()
        ):

            mask = (
                df["ml_label"] == label
            )

            class_X = X.loc[mask]

            class_missing = int(
                class_X.isna()
                .sum()
                .sum()
            )

            class_total = (
                class_X.shape[0]
                * class_X.shape[1]
            )

            class_missing_percent = (
                class_missing
                / class_total
                * 100
            )

            class_rows.append({
                "ml_label": label,
                "samples": class_X.shape[0],
                "missing_values": class_missing,
                "total_values": class_total,
                "missing_percent": class_missing_percent,
            })

        class_qc = pd.DataFrame(
            class_rows
        )

        print(
            class_qc.to_string(
                index=False
            )
        )

        class_qc.to_csv(
            OUT_DIR / "missingness_by_label.tsv",
            sep="\t",
            index=False,
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("QC SUMMARY")
    print("=" * 70)

    print(
        f"Samples: {len(df)}"
    )

    print(
        f"ELSA features: {len(feature_cols)}"
    )

    print(
        f"Overall missingness: "
        f"{missing_percent:.2f}%"
    )

    print(
        f"Regions with missing data: "
        f"{regions_with_missing}"
    )

    print(
        f"Completely missing regions: "
        f"{completely_missing_regions}"
    )

    print(
        f"Zero-variance regions: "
        f"{zero_variance_count}"
    )

    print(
        f"Invalid methylation values: "
        f"{invalid_count}"
    )

    print()
    print(
        f"QC files saved to: {OUT_DIR}"
    )


if __name__ == "__main__":
    main()