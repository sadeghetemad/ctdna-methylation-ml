from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RUN_TABLE = ROOT / "metadata/raw/SraRunTable.csv"
SUPPLEMENTARY = ROOT / "metadata/publication/ELSA_supplementary_tables.xlsx"

TARGET_DIR = ROOT / "metadata/targets"

CLINICAL_OUT = ROOT / "metadata/ELSA_clinical_labels.tsv"
MANIFEST_OUT = ROOT / "metadata/sample_manifest.tsv"
LABELED_MANIFEST_OUT = ROOT / "metadata/sample_manifest_labeled.tsv"
COHORT_OUT = ROOT / "metadata/primary_cohort_stage1_vs_control.tsv"
ACCESSION_OUT = ROOT / "metadata/primary_cohort_SRR_Acc_List.txt"

REGIONS_OUT = TARGET_DIR / "ELSA_plasma_2473_regions.tsv"
BED_OUT = TARGET_DIR / "ELSA_plasma_2473_hg19.bed"


def load_run_table():
    print("Loading SRA RunTable...")

    df = pd.read_csv(RUN_TABLE)

    required = [
        "Run",
        "BioSample",
        "Experiment",
        "Sample Name",
        "tissue",
        "Assay Type",
        "LibraryLayout",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing RunTable columns: {missing}"
        )

    print(f"  SRA runs: {len(df):,}")

    return df


def build_sample_manifest(run_table):
    print("\nBuilding sample manifest...")

    manifest = pd.DataFrame({
        "run_id": run_table["Run"],
        "sample_id": run_table["Sample Name"],
        "biosample": run_table["BioSample"],
        "experiment": run_table["Experiment"],
        "tissue": run_table["tissue"],
        "assay": run_table["Assay Type"],
        "layout": run_table["LibraryLayout"],
    })

    manifest.to_csv(
        MANIFEST_OUT,
        sep="\t",
        index=False,
    )

    print(f"  rows: {len(manifest):,}")
    print(f"  saved: {MANIFEST_OUT}")

    return manifest


def build_clinical_labels():
    print("\nReading Supplementary Table 7...")

    raw = pd.read_excel(
        SUPPLEMENTARY,
        sheet_name="Supplementary Table 7",
        header=None,
    )

    columns = [
        "sample_id",
        "predicted",
        "ground_truth",
        "stage",
        "cancer_type",
        "max_diameter_cm",
        "model_group",
        "age",
        "gender",
        "dna_input_ng",
        "dna_concentration_ng_ul",
    ]

    # Cancer section
    cancer = raw.iloc[2:311, :11].copy()
    cancer.columns = columns

    # Control section
    control = raw.iloc[313:, :11].copy()
    control.columns = columns

    clinical = pd.concat(
        [cancer, control],
        ignore_index=True,
    )

    # Keep actual BMD samples only.
    clinical = clinical[
        clinical["sample_id"]
        .astype(str)
        .str.match(r"^BMD\d+$", na=False)
    ].copy()

    # IMPORTANT:
    # labels come from ground_truth, not predicted.
    clinical["disease_status"] = clinical[
        "ground_truth"
    ].map({
        "tumor": "cancer",
        "normal": "control",
    })

    if clinical["disease_status"].isna().any():
        bad = clinical[
            clinical["disease_status"].isna()
        ][["sample_id", "ground_truth"]]

        raise ValueError(
            "Unexpected ground_truth values:\n"
            + bad.to_string(index=False)
        )

    if clinical["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate sample IDs found in clinical labels."
        )

    clinical.to_csv(
        CLINICAL_OUT,
        sep="\t",
        index=False,
    )

    print(f"  labelled samples: {len(clinical):,}")
    print(
        clinical["disease_status"]
        .value_counts()
        .to_string()
    )

    cancer_stage = (
        clinical[
            clinical["disease_status"] == "cancer"
        ]["stage"]
        .value_counts()
    )

    print("\n  Cancer stages:")
    print(cancer_stage.to_string())

    print(f"\n  saved: {CLINICAL_OUT}")

    return clinical


def build_labeled_manifest(manifest, clinical):
    print("\nJoining SRA metadata with clinical labels...")

    labeled = manifest.merge(
        clinical,
        on="sample_id",
        how="left",
        validate="many_to_one",
    )

    labeled.to_csv(
        LABELED_MANIFEST_OUT,
        sep="\t",
        index=False,
    )

    matched = labeled["disease_status"].notna().sum()

    print(f"  manifest rows: {len(labeled):,}")
    print(f"  labelled/matched runs: {matched:,}")
    print(f"  saved: {LABELED_MANIFEST_OUT}")

    return labeled


def build_primary_cohort(labeled):
    print("\nBuilding primary Stage-I vs control cohort...")

    cohort = labeled[
        (labeled["disease_status"] == "control")
        |
        (
            (labeled["disease_status"] == "cancer")
            &
            (labeled["stage"].isin(["IA", "IB"]))
        )
    ].copy()

    cohort["ml_label"] = (
        cohort["disease_status"]
        .map({
            "control": 0,
            "cancer": 1,
        })
        .astype(int)
    )

    if cohort["sample_id"].duplicated().any():
        raise ValueError(
            "Primary cohort contains duplicate biological samples."
        )

    if cohort["run_id"].duplicated().any():
        raise ValueError(
            "Primary cohort contains duplicate SRA runs."
        )

    cohort.to_csv(
        COHORT_OUT,
        sep="\t",
        index=False,
    )

    cohort["run_id"].to_csv(
        ACCESSION_OUT,
        index=False,
        header=False,
    )

    print(f"  total cohort: {len(cohort):,}")

    print("\n  Disease status:")
    print(
        cohort["disease_status"]
        .value_counts()
        .to_string()
    )

    print("\n  Stage/subtype:")
    print(
        cohort["stage"]
        .value_counts()
        .to_string()
    )

    print("\n  ML labels:")
    print(
        cohort["ml_label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(f"\n  saved: {COHORT_OUT}")
    print(f"  saved: {ACCESSION_OUT}")

    return cohort


def build_elsa_targets():
    print("\nReading Supplementary Table 5...")

    raw = pd.read_excel(
        SUPPLEMENTARY,
        sheet_name="Supplementary Table 5",
        header=None,
    )

    # First three columns contain the 2,473 plasma ELSA
    # classifier regions. Convert them to numeric genomic
    # coordinates and discard title/header rows.
    plasma = raw.iloc[:, :3].copy()

    plasma.columns = [
        "chrom",
        "start",
        "end",
    ]

    plasma["start"] = pd.to_numeric(
        plasma["start"],
        errors="coerce",
    )

    plasma["end"] = pd.to_numeric(
        plasma["end"],
        errors="coerce",
    )

    plasma = plasma.dropna(
        subset=["chrom", "start", "end"]
    ).copy()

    plasma["chrom"] = (
        plasma["chrom"]
        .astype(str)
        .str.strip()
    )

    plasma["start"] = plasma["start"].astype(int)
    plasma["end"] = plasma["end"].astype(int)

    # Publication coordinates are converted to BED:
    # BED start is zero-based; BED end remains unchanged.
    plasma["region_id"] = [
        f"ELSA_{i:04d}"
        for i in range(1, len(plasma) + 1)
    ]

    plasma.to_csv(
        REGIONS_OUT,
        sep="\t",
        index=False,
    )

    bed = pd.DataFrame({
        "chrom": plasma["chrom"],
        "start0": plasma["start"] - 1,
        "end": plasma["end"],
        "region_id": plasma["region_id"],
    })

    bed.to_csv(
        BED_OUT,
        sep="\t",
        index=False,
        header=False,
    )

    print(f"  plasma target regions: {len(bed):,}")
    print(f"  saved: {REGIONS_OUT}")
    print(f"  saved: {BED_OUT}")

    if len(bed) != 2473:
        raise ValueError(
            f"Expected 2473 plasma ELSA regions, got {len(bed)}"
        )


def validate_results(cohort):
    print("\nValidating expected cohort...")

    expected = {
        "total": 460,
        "cancer": 199,
        "control": 261,
        "IA": 157,
        "IB": 42,
        "Healthy": 246,
        "Benign": 15,
    }

    actual = {
        "total": len(cohort),
        "cancer": (
            cohort["disease_status"] == "cancer"
        ).sum(),
        "control": (
            cohort["disease_status"] == "control"
        ).sum(),
        "IA": (cohort["stage"] == "IA").sum(),
        "IB": (cohort["stage"] == "IB").sum(),
        "Healthy": (
            cohort["stage"] == "Healthy"
        ).sum(),
        "Benign": (
            cohort["stage"] == "Benign"
        ).sum(),
    }

    for key, expected_value in expected.items():

        actual_value = int(actual[key])

        status = (
            "OK"
            if actual_value == expected_value
            else "ERROR"
        )

        print(
            f"  {key:8s}: "
            f"{actual_value:4d} "
            f"(expected {expected_value}) "
            f"[{status}]"
        )

        if actual_value != expected_value:
            raise ValueError(
                f"{key}: expected {expected_value}, "
                f"got {actual_value}"
            )

    print("\nMetadata validation PASSED.")


def main():

    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_table = load_run_table()

    manifest = build_sample_manifest(
        run_table
    )

    clinical = build_clinical_labels()

    labeled = build_labeled_manifest(
        manifest,
        clinical,
    )

    cohort = build_primary_cohort(
        labeled
    )

    build_elsa_targets()

    validate_results(
        cohort
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()
