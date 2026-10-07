"""
Shared paths and helpers for the numbered pipeline scripts.

Every script can be run on its own (python scripts/NN_*.py); Python puts the
script directory on sys.path, so `from utils import ...` works from anywhere.
"""

from pathlib import Path
import csv
import os
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = ROOT / "logs"

# Threads for FastQC / Trim Galore / Bismark / samtools.
# Override without editing code:  CTDNA_THREADS=16 python scripts/03_process_custom.py
THREADS = int(os.environ.get("CTDNA_THREADS", "8"))

# How step 03 turns FASTQ into per-CpG methylation calls; main.py --mode sets it:
#   custom     : our own commands (03_process_custom.py)
#   nfcore     : nf-core/methylseq with Bismark (03_process_nfcore.py)
#   nfcore-gpu : nf-core/methylseq with bwa-meth on the GPU (Parabricks) and MethylDackel
# Override without editing code:  CTDNA_MODE=nfcore python scripts/04_features.py
MODES = ("custom", "nfcore", "nfcore-gpu")
MODE = os.environ.get("CTDNA_MODE", "custom")

if MODE not in MODES:
    raise ValueError(f"CTDNA_MODE must be one of {MODES}, got {MODE!r}")


# ============================================================
# Shared paths
# ============================================================

# Working cohort written by 01_data.py; every later step reads it.
# For the full cohort, point it at metadata/primary_cohort_stage1_vs_control.tsv
COHORT_FILE = ROOT / "metadata/test_cohort_10.tsv"
COHORT = COHORT_FILE.stem

TARGET_BED = ROOT / "metadata/targets/ELSA_plasma_2473_hg19.bed"
EXPECTED_REGIONS = 2473

FASTQ_DIR = ROOT / "data/fastq"
REF_DIR = ROOT / "reference/hg19"
FASTA = REF_DIR / "hg19.fa"

# Per-CpG coverage files in Bismark .cov format, <dir>/<run><suffix>, read by
# 04_features.py. The nf-core modes link (Bismark) or convert (MethylDackel) them.
COV_FILES = {
    "custom": (ROOT / "data/methylation", ".filtered.namesort.bismark.cov.gz"),
    "nfcore": (ROOT / "data/nfcore/methylation", ".bismark.cov.gz"),
    "nfcore-gpu": (ROOT / "data/nfcore-gpu/methylation", ".bismark.cov.gz"),
}

# nf-core results of the two nf-core modes
NFCORE_RESULTS = {
    "nfcore": ROOT / "results/nfcore_methylseq",
    "nfcore-gpu": ROOT / "results/nfcore_gpu_methylseq",
}

# Deduplicated (or duplicate-marked), coordinate-sorted, indexed BAMs,
# <dir>/<run><suffix>, read by 05_read_features.py
BAM_FILES = {
    "custom": (ROOT / "data/filtered", ".filtered.bam"),
    "nfcore": (NFCORE_RESULTS["nfcore"] / "bismark/deduplicated", ".deduplicated.sorted.bam"),
    "nfcore-gpu": (NFCORE_RESULTS["nfcore-gpu"] / "bwameth/deduplicated", ".markdup.sorted.bam"),
}

METH_DIR, COV_SUFFIX = COV_FILES[MODE]
BAM_DIR, BAM_SUFFIX = BAM_FILES[MODE]

# The first bp of every R2 are filled in without bisulfite conversion by the
# ELSA library prep and look ~90% methylated; every methylation caller
# ignores them (Bismark --ignore_r2, MethylDackel --nOT/--nOB, step 05).
IGNORE_R2_END = 10

# Read-level region features of 05_read_features.py -> the fragment count
# that supports each value; 07_ml_dataset.py masks values with low support.
READ_FEATURES = {
    "meth": "n_calls",                  # methylation fraction, artifact-free X_methylation
    "frac_meth": "n_frag",              # fragments with >= 80% methylated CpGs
    "frac_meth_short": "n_frag_short",  # the same, short fragments only
    "mhl": "n_frag",                    # methylation haplotype load
    "pdr": "n_frag4",                   # proportion of discordant reads
    "entropy": "n_frag4",               # 4-CpG pattern entropy
    "frac_short": "n_frag_all",         # short fragments among all fragments
}
READ_COUNTS = ("n_calls", "n_frag_all", "n_frag", "n_frag4", "n_frag_short")


def mode_dir(path):
    """Output directory of steps 04-07: custom keeps the path, the others add /<mode>."""
    path = ROOT / path
    return path if MODE == "custom" else path / MODE


FEATURES_DIR = mode_dir("features")
FEATURE_QC_DIR = mode_dir("results/feature_qc")
ML_DATA_DIR = mode_dir("notebooks/data")

# Sample-level features of 05_read_features.py (frag_*) and its QC columns
SAMPLE_FEATURES_FILE = FEATURES_DIR / f"{COHORT}_sample_features.tsv"


def read_feature_file(name):
    """Sample x region matrix of a READ_FEATURES / READ_COUNTS column."""
    return FEATURES_DIR / f"{COHORT}_read_{name}.tsv"


# ============================================================
# Cohort
# ============================================================

def cohort_runs():
    """Run IDs of the working cohort, in file order."""

    if not COHORT_FILE.exists():
        sys.exit(f"Missing cohort file: {COHORT_FILE}\nRun: python scripts/01_data.py")

    with COHORT_FILE.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    if not rows or "run_id" not in rows[0]:
        sys.exit(f"No run_id column in {COHORT_FILE}")

    return list(dict.fromkeys(r["run_id"].strip() for r in rows if r["run_id"].strip()))


def fastq_pair(run_id):
    return FASTQ_DIR / f"{run_id}_1.fastq.gz", FASTQ_DIR / f"{run_id}_2.fastq.gz"


def runs_with_fastq():
    """
    Cohort runs with both FASTQ files. Runs without them (e.g. a failed
    download) are left out with a warning; 04_features.py lists them.
    """

    run_ids = cohort_runs()
    present = [r for r in run_ids if all(p.exists() for p in fastq_pair(r))]
    missing = [r for r in run_ids if r not in present]

    if missing:
        print(f"WARNING: FASTQ files missing for {len(missing)} run(s), left out: "
              f"{', '.join(missing)}")
        print("         Run python scripts/01_data.py to download them.")

    if not present:
        sys.exit(f"No run of {COHORT_FILE.name} has both FASTQ files in {FASTQ_DIR}")

    return present


# ============================================================
# Running commands
# ============================================================

def _printable(cmd):
    return shlex.join(str(x) for x in cmd)


def _check(returncode, cmd, log_file):
    if returncode != 0:
        raise RuntimeError(
            f"exit code {returncode}: {_printable(cmd)}\n"
            f"         see log: {log_file}"
        )


def run(cmd, log_file):
    """
    Run a command, appending stdout/stderr to log_file.
    Raises RuntimeError on a non-zero exit code.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"  $ {_printable(cmd)}", flush=True)
    print(f"    log: {log_file}", flush=True)

    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {_printable(cmd)}\n")
        log.flush()

        result = subprocess.run(
            [str(x) for x in cmd],
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    _check(result.returncode, cmd, log_file)


def run_pipe(producer, consumer, log_file):
    """
    Run `producer | consumer` without a shell.

    Unlike `shell=True` without pipefail, a failure of the producer
    is detected even if the consumer exits successfully.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    printable = f"{_printable(producer)} | {_printable(consumer)}"
    print(f"  $ {printable}", flush=True)
    print(f"    log: {log_file}", flush=True)

    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {printable}\n")
        log.flush()

        first = subprocess.Popen(
            [str(x) for x in producer],
            stdout=subprocess.PIPE,
            stderr=log,
        )
        second = subprocess.run(
            [str(x) for x in consumer],
            stdin=first.stdout,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        first.stdout.close()
        first.wait()

    _check(first.returncode, producer, log_file)
    _check(second.returncode, consumer, log_file)


def all_exist(*paths):
    return all(Path(p).exists() for p in paths)


def tmp_path(path):
    """Temporary sibling path with the same extension: x.bam -> x.tmp.bam"""
    path = Path(path)
    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def finish(step, failed):
    """Print a summary and return a process exit code (1 if anything failed)."""
    print()
    print("=" * 70)

    if failed:
        print(f"{step}: {len(failed)} run(s) FAILED")
        for run_id, error in failed:
            print(f"  {run_id}: {error}")
        print("=" * 70)
        return 1

    print(f"{step}: all runs OK")
    print("=" * 70)
    return 0
