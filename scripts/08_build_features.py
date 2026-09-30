import argparse
import sys

import numpy as np
import pandas as pd

from utils import ROOT


# ============================================================
# Paths
# ============================================================

METH_DIR = ROOT / "data/methylation"

TARGET_FILE = (
    ROOT
    / "metadata/targets/ELSA_plasma_2473_hg19.bed"
)

COHORT_FILE = (
    ROOT
    / "metadata/test_cohort_10.tsv"
)

OUT_DIR = ROOT / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output names follow the cohort file: test_cohort_10 -> test_cohort_10_*.tsv
PREFIX = COHORT_FILE.stem

FEATURES_OUT = OUT_DIR / f"{PREFIX}_features.tsv"      # methylation fraction
COVERAGE_OUT = OUT_DIR / f"{PREFIX}_coverage.tsv"      # methylated + unmethylated calls
N_CPG_OUT = OUT_DIR / f"{PREFIX}_n_cpg.tsv"            # observed CpG sites
MISSING_OUT = OUT_DIR / f"{PREFIX}_missing_samples.tsv"

EXPECTED_REGIONS = 2473

META_COLUMNS = ["run_id", "sample_id", "ml_label"]


# ============================================================
# Chromosome normalization
# ============================================================

def normalize_chromosome(chrom):
    """
    Normalize chromosome names to UCSC-style naming.

    Examples:
        1     -> chr1
        X     -> chrX
        chr1  -> chr1

    The ELSA BED uses Ensembl-style names (1, 2, ...), while the
    UCSC hg19 reference and Bismark coverage use chr1, chr2, ...
    """

    chrom = str(chrom).strip()

    if not chrom.startswith("chr"):
        chrom = f"chr{chrom}"

    return chrom


# ============================================================
# Load target regions
# ============================================================

def load_targets():

    df = pd.read_csv(
        TARGET_FILE,
        sep="\t",
        header=None,
    )

    if df.shape[1] < 4:
        raise ValueError(
            f"Target BED must contain at least 4 columns. "
            f"Found {df.shape[1]}"
        )

    df = df.iloc[:, :4].copy()

    df.columns = [
        "chrom",
        "start",
        "end",
        "region_id",
    ]

    df["chrom"] = df["chrom"].map(normalize_chromosome)

    df["start"] = pd.to_numeric(df["start"], errors="raise").astype(np.int64)
    df["end"] = pd.to_numeric(df["end"], errors="raise").astype(np.int64)

    if df["region_id"].duplicated().any():

        duplicated = (
            df.loc[
                df["region_id"].duplicated(),
                "region_id",
            ]
            .tolist()
        )

        raise ValueError(
            f"Duplicate target region IDs found: "
            f"{duplicated[:10]}"
        )

    return df


# ============================================================
# Find Bismark coverage file
# ============================================================

def find_coverage_file(run_id):
    """
    Exact file name produced by 07_extract_methylation.py.

    No wildcard fallback: a pattern such as *SRR123*.cov.gz could
    silently pick up a different run (e.g. SRR1234).
    """

    path = METH_DIR / f"{run_id}.filtered.namesort.bismark.cov.gz"

    return path if path.exists() else None


# ============================================================
# Load Bismark coverage
# ============================================================

def load_coverage(path):
    """
    Bismark .cov columns:
      chrom, start (1-based), end, percent, methylated, unmethylated
    """

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        compression="gzip",
        usecols=[0, 1, 4, 5],
        names=["chrom", "start", "methylated", "unmethylated"],
        dtype={
            "chrom": str,
            "start": np.int64,
            "methylated": np.int64,
            "unmethylated": np.int64,
        },
    )

    df["chrom"] = df["chrom"].map(normalize_chromosome)

    return df


# ============================================================
# Validate chromosome compatibility
# ============================================================

def validate_chromosomes(
    coverage,
    targets,
    run_id,
):

    target_chroms = set(targets["chrom"].unique())
    coverage_chroms = set(coverage["chrom"].unique())

    common_chroms = target_chroms & coverage_chroms

    if not common_chroms:

        raise ValueError(
            f"No matching chromosomes between "
            f"target BED and coverage for {run_id}.\n"
            f"Target chromosomes: "
            f"{sorted(target_chroms)}\n"
            f"Coverage chromosomes: "
            f"{sorted(coverage_chroms)}"
        )

    print(f"  Matching chromosomes: {len(common_chroms)}")


# ============================================================
# Aggregate CpG methylation into ELSA regions
# ============================================================

def aggregate_sample(
    coverage,
    targets,
):
    """
    Per region: number of CpG sites, methylated / unmethylated calls,
    and the coverage-weighted methylation fraction.

    Coordinate systems
      BED:              start 0-based, end exclusive
      Bismark coverage: CpG position 1-based
    therefore
      BED start + 1 <= CpG position <= BED end

    Implementation: per chromosome, sort CpG positions once, take
    cumulative sums of the counts and look up each region's first/last
    CpG with binary search. Overlapping regions are handled correctly.
    """

    n = len(targets)

    n_cpg = np.zeros(n, dtype=np.int64)
    methylated = np.zeros(n, dtype=np.int64)
    unmethylated = np.zeros(n, dtype=np.int64)

    coverage_by_chr = {
        chrom: group.sort_values("start")
        for chrom, group in coverage.groupby("chrom", sort=False)
    }

    for chrom, idx in targets.groupby("chrom", sort=False).indices.items():

        if chrom not in coverage_by_chr:
            continue

        c = coverage_by_chr[chrom]

        pos = c["start"].to_numpy()
        cum_m = np.concatenate([[0], np.cumsum(c["methylated"].to_numpy())])
        cum_u = np.concatenate([[0], np.cumsum(c["unmethylated"].to_numpy())])

        region_start_1based = targets["start"].to_numpy()[idx] + 1
        region_end_1based = targets["end"].to_numpy()[idx]

        lo = np.searchsorted(pos, region_start_1based, side="left")
        hi = np.searchsorted(pos, region_end_1based, side="right")

        n_cpg[idx] = hi - lo
        methylated[idx] = cum_m[hi] - cum_m[lo]
        unmethylated[idx] = cum_u[hi] - cum_u[lo]

    total = methylated + unmethylated

    # Coverage-weighted methylation fraction.
    # Do NOT average percent_methylated directly:
    # CpGs may have different read depths.
    fraction = np.full(n, np.nan)
    covered = total > 0
    fraction[covered] = methylated[covered] / total[covered]

    return pd.DataFrame({
        "n_cpg": n_cpg,
        "coverage": total,
        "fraction": fraction,
    })


# ============================================================
# Main
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description="Aggregate Bismark CpG coverage into ELSA region features."
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any cohort sample has no coverage file.",
    )
    return p.parse_args()


def main():

    args = parse_args()

    # --------------------------------------------------------
    # Load target regions
    # --------------------------------------------------------

    targets = load_targets()

    print(f"Target regions: {len(targets)}")

    if len(targets) != EXPECTED_REGIONS:

        raise ValueError(
            f"Expected {EXPECTED_REGIONS} target regions, "
            f"found {len(targets)}"
        )

    print(
        "Target chromosomes:",
        sorted(targets["chrom"].unique()),
    )

    region_ids = targets["region_id"].tolist()

    # --------------------------------------------------------
    # Load selected cohort
    # --------------------------------------------------------

    cohort = pd.read_csv(
        COHORT_FILE,
        sep="\t",
    )

    if "run_id" not in cohort.columns:

        raise ValueError(
            "Cohort file does not contain "
            "'run_id' column."
        )

    print(f"Cohort samples requested: {len(cohort)}")

    meta_rows = []
    fraction_rows = []
    coverage_rows = []
    n_cpg_rows = []
    missing_samples = []

    # --------------------------------------------------------
    # Process each sample
    # --------------------------------------------------------

    for _, sample in cohort.iterrows():

        run_id = str(sample["run_id"])

        coverage_file = find_coverage_file(run_id)

        if coverage_file is None:

            print(f"\n[MISSING] No coverage file: {run_id}")

            missing_samples.append({
                c: sample[c] for c in META_COLUMNS if c in cohort.columns
            })

            continue

        print()
        print(f"[FEATURES] {run_id}")
        print(f"  Coverage file: {coverage_file.name}")

        coverage = load_coverage(coverage_file)

        print(f"  CpG records: {len(coverage):,}")

        validate_chromosomes(
            coverage,
            targets,
            run_id,
        )

        region = aggregate_sample(
            coverage,
            targets,
        )

        covered_regions = int(region["fraction"].notna().sum())
        missing_regions = len(region) - covered_regions

        print(
            f"  Covered ELSA regions: "
            f"{covered_regions}/{len(region)}"
        )

        print(
            f"  Missing ELSA regions: "
            f"{missing_regions}/{len(region)} "
            f"({missing_regions / len(region) * 100:.2f}%)"
        )

        print(
            f"  Median region depth: "
            f"{region['coverage'].median():.0f} calls"
        )

        # ----------------------------------------------------
        # Critical sample-level validation
        # ----------------------------------------------------

        if covered_regions == 0:

            raise RuntimeError(
                f"{run_id} has zero covered "
                f"ELSA regions. "
                f"Check genome build, chromosome "
                f"naming, and coordinate systems."
            )

        meta_rows.append({
            c: sample[c] for c in META_COLUMNS if c in cohort.columns
        })

        fraction_rows.append(region["fraction"].to_numpy())
        coverage_rows.append(region["coverage"].to_numpy())
        n_cpg_rows.append(region["n_cpg"].to_numpy())

    # --------------------------------------------------------
    # Missing samples
    # --------------------------------------------------------

    if missing_samples:

        missing_df = pd.DataFrame(missing_samples)
        missing_df.to_csv(MISSING_OUT, sep="\t", index=False)

        print()
        print("!" * 70)
        print(
            f"WARNING: {len(missing_samples)}/{len(cohort)} cohort "
            f"samples have NO coverage file and are NOT in the matrix:"
        )
        print(missing_df.to_string(index=False))
        print(f"Listed in: {MISSING_OUT}")
        print("!" * 70)

        if args.strict:
            raise RuntimeError(
                "Missing samples with --strict. "
                "Run steps 001-07 for them first."
            )

    else:
        MISSING_OUT.unlink(missing_ok=True)

    # --------------------------------------------------------
    # Build matrices
    # --------------------------------------------------------

    if not meta_rows:

        raise RuntimeError(
            "No samples were processed. "
            "No feature matrix can be created."
        )

    meta = pd.DataFrame(meta_rows)

    def matrix(rows):
        values = pd.DataFrame(np.vstack(rows), columns=region_ids)
        return pd.concat([meta, values], axis=1)

    feature_matrix = matrix(fraction_rows)
    coverage_matrix = matrix(coverage_rows)
    n_cpg_matrix = matrix(n_cpg_rows)

    # --------------------------------------------------------
    # Matrix-level validation
    # --------------------------------------------------------

    missing = int(feature_matrix[region_ids].isna().sum().sum())
    total = len(feature_matrix) * len(region_ids)
    missing_fraction = missing / total

    # Absolutely do not silently write an all-NaN matrix.
    if missing_fraction == 1.0:

        raise RuntimeError(
            "All ELSA features are missing "
            "for all samples. "
            "Feature matrix will NOT be written. "
            "Check chromosome naming, genome build, "
            "and genomic coordinates."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    feature_matrix.to_csv(FEATURES_OUT, sep="\t", index=False)
    coverage_matrix.to_csv(COVERAGE_OUT, sep="\t", index=False)
    n_cpg_matrix.to_csv(N_CPG_OUT, sep="\t", index=False)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FEATURE MATRIX COMPLETE")
    print("=" * 70)

    print(f"Samples: {len(feature_matrix)} (of {len(cohort)} in cohort)")

    if "ml_label" in feature_matrix.columns:
        counts = feature_matrix["ml_label"].value_counts().sort_index()
        print(
            "Labels:  "
            + ", ".join(f"{k}={v}" for k, v in counts.items())
        )

    print(f"Features: {len(region_ids)}")

    print(
        f"Missing values: "
        f"{missing}/{total} "
        f"({missing_fraction * 100:.2f}%)"
    )

    print(
        f"Observed values: "
        f"{total - missing}/{total} "
        f"({(1 - missing_fraction) * 100:.2f}%)"
    )

    print(f"Saved: {FEATURES_OUT}")
    print(f"Saved: {COVERAGE_OUT}")
    print(f"Saved: {N_CPG_OUT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
