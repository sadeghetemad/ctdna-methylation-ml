"""
Step 01: everything before the reads are processed.

    metadata : SRA run table + publication tables -> labels, manifests,
               the Stage I vs. control cohort and the ELSA target BED
    cohort   : a balanced, reproducible working cohort (COHORT_FILE)
    download : paired FASTQ files of the working cohort from ENA,
               checked by size and MD5

Examples:
    python scripts/01_data.py                       # all stages
    python scripts/01_data.py --stages download     # only the download
    python scripts/01_data.py --strict              # fail on a failed download
"""

import argparse
import hashlib
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

from utils import (
    ROOT, LOG_DIR, COHORT_FILE, COHORT, TARGET_BED, EXPECTED_REGIONS,
    FASTQ_DIR, cohort_runs,
)


STAGES = ("metadata", "cohort", "download")

# Inputs (tracked in git)
RUN_TABLE = ROOT / "metadata/raw/SraRunTable.csv"
SUPPLEMENTARY = ROOT / "metadata/publication/ELSA_supplementary_tables.xlsx"

# metadata outputs
CLINICAL_OUT = ROOT / "metadata/ELSA_clinical_labels.tsv"
MANIFEST_OUT = ROOT / "metadata/sample_manifest.tsv"
LABELED_MANIFEST_OUT = ROOT / "metadata/sample_manifest_labeled.tsv"
PRIMARY_COHORT = ROOT / "metadata/primary_cohort_stage1_vs_control.tsv"
PRIMARY_ACCESSIONS = ROOT / "metadata/primary_cohort_SRR_Acc_List.txt"
REGIONS_OUT = TARGET_BED.parent / "ELSA_plasma_2473_regions.tsv"

# cohort: N_PER_CLASS cancer + N_PER_CLASS control runs
N_PER_CLASS = 5
SEED = 42
COHORT_ACCESSIONS = COHORT_FILE.with_name(f"{COHORT}_SRR.txt")

# download
ENA_API = "https://www.ebi.ac.uk/ena/portal/api/filereport"
FAILED_DOWNLOADS = LOG_DIR / "failed_downloads.tsv"

# Counts of the primary cohort in the publication
EXPECTED_COHORT = {
    "total": 460, "cancer": 199, "control": 261,
    "IA": 157, "IB": 42, "Healthy": 246, "Benign": 15,
}


def parse_args():

    p = argparse.ArgumentParser(
        description="Metadata, working cohort and FASTQ download."
    )
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES),
                   help="Stages to run, always in pipeline order (default: all).")
    p.add_argument("--strict", action="store_true",
                   help="Exit with an error if any download fails "
                        "(default: warn; later steps leave the run out).")
    return p.parse_args()


def save(df, path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, **kwargs)
    print(f"  saved: {path.relative_to(ROOT)}")


def counts(series):
    return series.value_counts().to_string()


# ============================================================
# metadata
# ============================================================

def load_run_table():

    df = pd.read_csv(RUN_TABLE)

    required = ["Run", "BioSample", "Experiment", "Sample Name",
                "tissue", "Assay Type", "LibraryLayout"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing RunTable columns: {missing}")

    print(f"SRA runs: {len(df):,}")

    return pd.DataFrame({
        "run_id": df["Run"],
        "sample_id": df["Sample Name"],
        "biosample": df["BioSample"],
        "experiment": df["Experiment"],
        "tissue": df["tissue"],
        "assay": df["Assay Type"],
        "layout": df["LibraryLayout"],
    })


def load_clinical_labels():
    """Supplementary Table 7: cancer rows 2-310, control rows 313-."""

    raw = pd.read_excel(SUPPLEMENTARY, sheet_name="Supplementary Table 7", header=None)

    columns = [
        "sample_id", "predicted", "ground_truth", "stage", "cancer_type",
        "max_diameter_cm", "model_group", "age", "gender",
        "dna_input_ng", "dna_concentration_ng_ul",
    ]

    cancer = raw.iloc[2:311, :11].copy()
    control = raw.iloc[313:, :11].copy()
    cancer.columns = control.columns = columns

    clinical = pd.concat([cancer, control], ignore_index=True)

    # Keep actual BMD samples only
    clinical = clinical[
        clinical["sample_id"].astype(str).str.match(r"^BMD\d+$", na=False)
    ].copy()

    # Labels come from ground_truth, NOT from predicted (the published classifier)
    clinical["disease_status"] = clinical["ground_truth"].map(
        {"tumor": "cancer", "normal": "control"}
    )

    if clinical["disease_status"].isna().any():
        bad = clinical.loc[clinical["disease_status"].isna(), ["sample_id", "ground_truth"]]
        raise ValueError("Unexpected ground_truth values:\n" + bad.to_string(index=False))

    if clinical["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs found in clinical labels.")

    print(f"Labelled samples: {len(clinical):,}")
    print(counts(clinical["disease_status"]))
    print("Cancer stages:")
    print(counts(clinical.loc[clinical["disease_status"] == "cancer", "stage"]))

    return clinical


def build_primary_cohort(labeled):
    """All controls and Stage I (IA, IB) cancers."""

    is_control = labeled["disease_status"] == "control"
    is_stage1 = (labeled["disease_status"] == "cancer") & labeled["stage"].isin(["IA", "IB"])

    cohort = labeled[is_control | is_stage1].copy()
    cohort["ml_label"] = cohort["disease_status"].map({"control": 0, "cancer": 1}).astype(int)

    if cohort["sample_id"].duplicated().any():
        raise ValueError("Primary cohort contains duplicate biological samples.")
    if cohort["run_id"].duplicated().any():
        raise ValueError("Primary cohort contains duplicate SRA runs.")

    print(f"Primary cohort: {len(cohort):,}")
    print(counts(cohort["stage"]))

    return cohort


def validate_primary_cohort(cohort):

    status, stage = cohort["disease_status"], cohort["stage"]
    actual = {
        "total": len(cohort),
        "cancer": (status == "cancer").sum(),
        "control": (status == "control").sum(),
        **{s: (stage == s).sum() for s in ("IA", "IB", "Healthy", "Benign")},
    }

    for key, expected in EXPECTED_COHORT.items():
        ok = int(actual[key]) == expected
        print(f"  {key:8s}: {int(actual[key]):4d} (expected {expected}) [{'OK' if ok else 'ERROR'}]")
        if not ok:
            raise ValueError(f"{key}: expected {expected}, got {int(actual[key])}")


def build_targets():
    """
    Supplementary Table 5, first three columns: the plasma ELSA regions.
    Publication coordinates are converted to BED: start - 1, end unchanged.
    """

    raw = pd.read_excel(SUPPLEMENTARY, sheet_name="Supplementary Table 5", header=None)

    plasma = raw.iloc[:, :3].copy()
    plasma.columns = ["chrom", "start", "end"]

    # Numeric coordinates only; drops the title and header rows
    plasma["start"] = pd.to_numeric(plasma["start"], errors="coerce")
    plasma["end"] = pd.to_numeric(plasma["end"], errors="coerce")
    plasma = plasma.dropna(subset=["chrom", "start", "end"]).copy()

    plasma["chrom"] = plasma["chrom"].astype(str).str.strip()
    plasma["start"] = plasma["start"].astype(int)
    plasma["end"] = plasma["end"].astype(int)
    plasma["region_id"] = [f"ELSA_{i:04d}" for i in range(1, len(plasma) + 1)]

    if len(plasma) != EXPECTED_REGIONS:
        raise ValueError(f"Expected {EXPECTED_REGIONS} plasma ELSA regions, got {len(plasma)}")

    print(f"Target regions: {len(plasma):,}")

    bed = pd.DataFrame({
        "chrom": plasma["chrom"],
        "start0": plasma["start"] - 1,
        "end": plasma["end"],
        "region_id": plasma["region_id"],
    })

    return plasma, bed


def prepare_metadata():

    manifest = load_run_table()
    clinical = load_clinical_labels()

    labeled = manifest.merge(clinical, on="sample_id", how="left", validate="many_to_one")
    print(f"Runs matched to a label: {labeled['disease_status'].notna().sum():,}"
          f" of {len(labeled):,}")

    cohort = build_primary_cohort(labeled)
    validate_primary_cohort(cohort)
    plasma, bed = build_targets()

    save(manifest, MANIFEST_OUT)
    save(clinical, CLINICAL_OUT)
    save(labeled, LABELED_MANIFEST_OUT)
    save(cohort, PRIMARY_COHORT)
    save(cohort["run_id"], PRIMARY_ACCESSIONS, header=False)
    save(plasma, REGIONS_OUT)
    save(bed, TARGET_BED, header=False)


# ============================================================
# cohort
# ============================================================

def select_cohort():

    if not PRIMARY_COHORT.exists():
        sys.exit(f"Missing {PRIMARY_COHORT}; run the metadata stage first.")

    df = pd.read_csv(PRIMARY_COHORT, sep="\t")

    cancer = df[df["ml_label"] == 1].sample(n=N_PER_CLASS, random_state=SEED)
    control = df[df["ml_label"] == 0].sample(n=N_PER_CLASS, random_state=SEED)

    # Shuffled, but reproducible
    selected = (
        pd.concat([cancer, control], ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    print(selected[["sample_id", "run_id", "disease_status", "stage", "ml_label"]]
          .to_string(index=False))

    save(selected, COHORT_FILE)
    save(selected["run_id"], COHORT_ACCESSIONS, header=False)


# ============================================================
# download
# ============================================================

def md5sum(path, chunk_size=8 * 1024 * 1024):

    md5 = hashlib.md5()

    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)

    return md5.hexdigest()


def ena_fastq_files(run_id):
    """URL, file name, MD5 and size of both FASTQ files of a run."""

    params = {
        "accession": run_id,
        "result": "read_run",
        "fields": "run_accession,fastq_ftp,fastq_md5,fastq_bytes",
        "format": "tsv",
    }
    url = ENA_API + "?" + urllib.parse.urlencode(params)

    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8")

    log_file = LOG_DIR / "ena" / f"{run_id}_ena.tsv"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(text)

    lines = text.strip().splitlines()

    if len(lines) < 2:
        raise RuntimeError(f"ENA returned no FASTQ information for {run_id}")

    row = dict(zip(lines[0].split("\t"), lines[1].split("\t")))
    urls = row["fastq_ftp"].split(";")

    if len(urls) != 2:
        raise RuntimeError(f"expected 2 paired FASTQ files, ENA returned {len(urls)}")

    files = []

    for ftp, md5, size in zip(urls, row["fastq_md5"].split(";"), row["fastq_bytes"].split(";")):
        ftp = ftp.strip()
        url = ftp if ftp.startswith("http") else "https://" + ftp
        files.append({
            "url": url,
            "path": FASTQ_DIR / Path(urllib.parse.urlparse(url).path).name,
            "md5": md5.strip(),
            "bytes": int(size),
        })

    return files


def download(url, destination, expected_bytes):
    """Download to <file>.part and rename it only when the size is right."""

    if destination.exists():
        size = destination.stat().st_size

        if size == expected_bytes:
            print(f"[EXISTS] {destination.name} ({size / 1024**3:.2f} GB)")
            return

        print(f"[INCOMPLETE] {destination.name}: {size} / {expected_bytes} bytes")
        destination.unlink()

    print(f"[DOWNLOAD] {destination.name}")
    print(f"           {url}")

    part = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ctDNA-ELSA-pipeline/1.0"})

    with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as out:

        total = 0
        last_print = time.time()

        while chunk := response.read(8 * 1024 * 1024):
            out.write(chunk)
            total += len(chunk)

            if time.time() - last_print >= 5:
                print(f"           {total / 1024**3:.2f} GB "
                      f"({total / expected_bytes * 100:.1f}%)", flush=True)
                last_print = time.time()

    size = part.stat().st_size

    if size != expected_bytes:
        part.unlink()
        raise RuntimeError(f"size mismatch for {destination.name}: "
                           f"expected {expected_bytes}, downloaded {size}")

    part.rename(destination)
    print(f"[OK] {destination.name} ({size / 1024**3:.2f} GB)")


def verify_md5(path, expected):

    actual = md5sum(path)

    if actual.lower() != expected.lower():
        # Delete it, or the size check would accept it again on every rerun
        path.unlink()
        raise RuntimeError(f"MD5 failed for {path.name}: expected {expected}, got {actual}")

    print(f"[MD5 OK] {path.name}")


def download_fastq(strict):

    FASTQ_DIR.mkdir(parents=True, exist_ok=True)

    run_ids = cohort_runs()
    failed = []

    for n, run_id in enumerate(run_ids, start=1):

        print(f"\n[{n}/{len(run_ids)}] {run_id}")

        try:
            for f in ena_fastq_files(run_id):
                download(f["url"], f["path"], f["bytes"])
                verify_md5(f["path"], f["md5"])

        except Exception as e:
            print(f"[FAILED] {run_id}: {e}")
            failed.append({"run_id": run_id, "error": str(e)})

    print(f"\nDownloaded and verified: {len(run_ids) - len(failed)}/{len(run_ids)} runs")

    if not failed:
        FAILED_DOWNLOADS.unlink(missing_ok=True)
        return 0

    save(pd.DataFrame(failed), FAILED_DOWNLOADS)

    for f in failed:
        print(f"  FAILED {f['run_id']}: {f['error']}")

    if strict:
        return 1

    print("WARNING: later steps leave the failed runs out (use --strict to stop here).")
    return 0


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()
    returncode = 0

    for stage in STAGES:

        if stage not in args.stages:
            continue

        print()
        print("=" * 70)
        print(f"01 data: {stage}")
        print("=" * 70)

        if stage == "metadata":
            prepare_metadata()
        elif stage == "cohort":
            select_cohort()
        elif stage == "download":
            returncode = download_fastq(args.strict)

    return returncode


if __name__ == "__main__":
    sys.exit(main())
