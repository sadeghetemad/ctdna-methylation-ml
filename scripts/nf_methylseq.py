"""
nfcore mode: run nf-core/methylseq in place of steps 01-07.

    1. Writes the samplesheet from the cohort and the FASTQ files of 001
       (runs without both FASTQ files are left out with a warning).
    2. Writes the ELSA target BED with UCSC chromosome names (chr1, ...).
    3. Runs nf-core/methylseq (FastQC, Trim Galore, Bismark, dedup,
       methylation extraction, MultiQC) with config/methylseq_*.
    4. Links each run's .bismark.cov.gz to data/nfcore/methylation/,
       where 08_build_features.py reads it in nfcore mode.

Usually run through main.py:
    python main.py --mode nfcore

On its own:
    python scripts/nf_methylseq.py --profile docker
    python scripts/nf_methylseq.py --dry-run
    python scripts/nf_methylseq.py --collect-only
"""

import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys

from utils import ROOT, NFCORE_COV


PIPELINE = "nf-core/methylseq"
REVISION = "4.2.0"

COHORT_FILE = ROOT / "metadata/test_cohort_10.tsv"
FASTQ_DIR = ROOT / "data/fastq"
TARGET_FILE = ROOT / "metadata/targets/ELSA_plasma_2473_hg19.bed"
REF_DIR = ROOT / "reference/hg19"
FASTA = REF_DIR / "hg19.fa"

PARAMS_FILE = ROOT / "config/methylseq_params.yaml"
CONFIG_FILE = ROOT / "config/methylseq.config"

# Nextflow state, inputs and work files (large) live under data/nfcore/
NF_DIR = ROOT / "data/nfcore"
SAMPLESHEET = NF_DIR / "inputs/samplesheet.csv"
TARGETS_UCSC = NF_DIR / "inputs/ELSA_plasma_2473_hg19.ucsc.bed"
WORK_DIR = NF_DIR / "work"

# Where 08_build_features.py looks in nfcore mode
COV_DIR, COV_SUFFIX = NFCORE_COV

# nf-core/methylseq 4.2.0 includes conf/aws/batch/nextflow.config, a file
# missing from the release. Nextflow >= 25.10 parses configs strictly and
# fails on it; the legacy parser only reads the profiles in use.
NXF_ENV = {"NXF_SYNTAX_PARSER": "v1"}

# nf-core results (MultiQC, BAMs, methylation calls)
OUT_DIR = ROOT / "results/nfcore_methylseq"


def parse_args():

    p = argparse.ArgumentParser(
        description="Run nf-core/methylseq for the ELSA cohort."
    )
    p.add_argument("--profile", default="docker",
                   help="Nextflow profile: docker, singularity, apptainer, "
                        "conda, ... (default: %(default)s).")
    p.add_argument("--revision", default=REVISION,
                   help="Pipeline version (default: %(default)s).")
    p.add_argument("--build-index", action="store_true",
                   help="Let nf-core build the Bismark index instead of "
                        "using reference/hg19/Bisulfite_Genome.")
    p.add_argument("--no-resume", action="store_true",
                   help="Start from scratch instead of -resume.")
    p.add_argument("--dry-run", action="store_true",
                   help="Write the inputs and print the command only.")
    p.add_argument("--collect-only", action="store_true",
                   help="Skip Nextflow; only link the coverage files.")
    return p.parse_args()


# ============================================================
# Inputs
# ============================================================

def read_run_ids():

    if not COHORT_FILE.exists():
        sys.exit(f"Missing cohort file: {COHORT_FILE}")

    with COHORT_FILE.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    if not rows or "run_id" not in rows[0]:
        sys.exit(f"No run_id column in {COHORT_FILE}")

    return [row["run_id"].strip() for row in rows if row["run_id"].strip()]


def fastq_pair(run_id):
    return FASTQ_DIR / f"{run_id}_1.fastq.gz", FASTQ_DIR / f"{run_id}_2.fastq.gz"


def runs_with_fastq(run_ids):
    """
    Cohort runs with both FASTQ files. Runs without them (e.g. a failed
    download in 001) are left out with a warning, as 08 does without --strict.
    """

    present = [r for r in run_ids if all(p.exists() for p in fastq_pair(r))]
    missing = [r for r in run_ids if r not in present]

    if missing:
        print(f"WARNING: FASTQ files missing for {len(missing)} run(s), left out: "
              f"{', '.join(missing)}")
        print("         Run python scripts/001_download_fastq.py to add them.")

    if not present:
        sys.exit(f"No run of {COHORT_FILE.name} has both FASTQ files in {FASTQ_DIR}")

    return present


def write_samplesheet(run_ids):

    rows = []

    for run_id in run_ids:
        r1, r2 = fastq_pair(run_id)
        rows.append({"sample": run_id, "fastq_1": r1, "fastq_2": r2, "genome": ""})

    SAMPLESHEET.parent.mkdir(parents=True, exist_ok=True)

    with SAMPLESHEET.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "fastq_1", "fastq_2", "genome"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Samplesheet: {SAMPLESHEET} ({len(rows)} samples)")


def write_ucsc_targets():
    """The panel BED uses 1, 2, ...; hg19.fa uses chr1, chr2, ..."""

    if not TARGET_FILE.exists():
        sys.exit(
            f"Missing target file: {TARGET_FILE}\n"
            "Run: python scripts/0_prepare_metadata.py"
        )

    TARGETS_UCSC.parent.mkdir(parents=True, exist_ok=True)

    with TARGET_FILE.open() as src, TARGETS_UCSC.open("w") as dst:
        for line in src:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue

            fields = line.rstrip("\n").split("\t")
            chrom = fields[0]

            if chrom in ("MT", "M"):
                chrom = "chrM"
            elif not chrom.startswith("chr"):
                chrom = f"chr{chrom}"

            dst.write("\t".join([chrom] + fields[1:]) + "\n")

    print(f"Targets:     {TARGETS_UCSC}")


# ============================================================
# Nextflow
# ============================================================

def nextflow_command(args):

    if not FASTA.exists():
        sys.exit(
            f"Missing reference: {FASTA}\n"
            "Run: python scripts/04_prepare_references.py"
        )

    cmd = [
        "nextflow", "run", PIPELINE,
        "-r", args.revision,
        "-profile", args.profile,
        "-params-file", PARAMS_FILE,
        "-c", CONFIG_FILE,
        "-work-dir", WORK_DIR,
        "--input", SAMPLESHEET,
        "--outdir", OUT_DIR,
        "--fasta", FASTA,
        "--target_regions_file", TARGETS_UCSC,
    ]

    if not args.build_index and (REF_DIR / "Bisulfite_Genome").exists():
        cmd += ["--bismark_index", REF_DIR]
    else:
        cmd += ["--save_reference"]

    if not args.no_resume:
        cmd.append("-resume")

    return [str(x) for x in cmd]


# ============================================================
# Collect coverage files for 08_build_features.py
# ============================================================

def collect_coverage(run_ids):
    """
    Link <run>*.bismark.cov.gz from the nf-core results to
    data/nfcore/methylation/<run>.bismark.cov.gz.

    nf-core adds suffixes to the sample name (e.g. _1_val_1_bismark_bt2_pe
    .deduplicated), so a run matches only when its ID is followed by
    "_" or "."; SRR123 never matches SRR1234.
    """

    found = sorted(OUT_DIR.rglob(f"*{COV_SUFFIX}"))
    COV_DIR.mkdir(parents=True, exist_ok=True)

    missing = []
    ambiguous = []

    for run_id in run_ids:
        matches = [
            p for p in found
            if p.name.startswith((f"{run_id}_", f"{run_id}."))
        ]

        if not matches:
            missing.append(run_id)
            continue

        if len(matches) > 1:
            ambiguous.append((run_id, matches))
            continue

        link = COV_DIR / f"{run_id}{COV_SUFFIX}"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(matches[0].resolve())

    linked = len(run_ids) - len(missing) - len(ambiguous)
    print(f"\nCoverage files linked: {linked}/{len(run_ids)} -> {COV_DIR}")

    for run_id in missing:
        print(f"  MISSING   {run_id}")
    for run_id, matches in ambiguous:
        print(f"  AMBIGUOUS {run_id}: {', '.join(p.name for p in matches)}")

    return 1 if missing or ambiguous else 0


def check_docker(profile):
    """Fail early when the docker profile is used but Docker is not usable."""

    if "docker" not in profile.split(","):
        return

    if shutil.which("docker") is None:
        sys.exit("docker not found. Install it: sudo apt-get install -y docker.io")

    result = subprocess.run(["docker", "info"], capture_output=True, text=True)

    if result.returncode != 0:
        sys.exit(
            "Docker is installed but not usable by this user:\n"
            f"  {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ''}\n"
            "Add the user to the docker group and log in again:\n"
            "  sudo usermod -aG docker $USER\n"
            "In a shell that was open before (e.g. VS Code), run instead:\n"
            '  sg docker -c "python main.py --mode nfcore"'
        )


def main():

    args = parse_args()
    run_ids = runs_with_fastq(read_run_ids())

    if not args.collect_only:

        write_samplesheet(run_ids)
        write_ucsc_targets()
        cmd = nextflow_command(args)

        print()
        print("$ " + " ".join(f"{k}={v}" for k, v in NXF_ENV.items()) + " " + shlex.join(cmd))

        if args.dry_run:
            return 0

        if shutil.which("nextflow") is None:
            sys.exit(
                "nextflow not found. Install it into the environment:\n"
                "  conda env update -f environment.yml"
            )

        check_docker(args.profile)

        # Nextflow writes .nextflow/ and .nextflow.log to the launch directory
        NF_DIR.mkdir(parents=True, exist_ok=True)
        env = {**NXF_ENV, **os.environ}    # a value set by the user wins
        returncode = subprocess.run(cmd, cwd=NF_DIR, env=env).returncode

        if returncode != 0:
            print(f"\nnf-core/methylseq failed (exit code {returncode}).")
            print(f"See {NF_DIR / '.nextflow.log'}; rerun to resume.")
            return returncode

    return collect_coverage(run_ids)


if __name__ == "__main__":
    sys.exit(main())
