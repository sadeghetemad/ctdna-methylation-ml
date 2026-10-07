"""
Step 03, nfcore and nfcore-gpu modes: nf-core/methylseq in place of
03_process_custom.py.

    nfcore      Bismark + Bowtie2 on the CPU, Bismark methylation extractor
    nfcore-gpu  bwa-meth on the GPU (Parabricks fq2bammeth), MethylDackel;
                needs an NVIDIA GPU and its container runtime

    1. Writes the samplesheet from the cohort runs with FASTQ files
       (runs without them are left out with a warning).
    2. Writes the ELSA target BED with UCSC chromosome names (chr1, ...).
    3. Runs nf-core/methylseq (FastQC, Trim Galore, alignment, dedup,
       methylation calls, MultiQC) with config/methylseq_*.
    4. Puts each run's CpG calls in Bismark .cov format into
       data/<mode>/methylation/, where 04_features.py reads them:
       a link to the Bismark file, or a conversion of the MethylDackel
       bedGraph (0-based) to .cov (1-based).

The mode comes from CTDNA_MODE (main.py --mode sets it); run on its own,
nfcore is the default.

Usually run through main.py:
    python main.py --mode nfcore
    python main.py --mode nfcore-gpu

On its own:
    python scripts/03_process_nfcore.py --profile docker
    python scripts/03_process_nfcore.py --dry-run
    CTDNA_MODE=nfcore-gpu python scripts/03_process_nfcore.py --dry-run
    python scripts/03_process_nfcore.py --collect-only
"""

import argparse
import csv
import gzip
import os
import shlex
import shutil
import subprocess
import sys

from utils import (
    ROOT, MODE, TARGET_BED, REF_DIR, FASTA, COV_FILES, NFCORE_RESULTS,
    fastq_pair, runs_with_fastq,
)


PIPELINE = "nf-core/methylseq"
REVISION = "4.2.0"

NF_MODE = MODE if MODE in NFCORE_RESULTS else "nfcore"
GPU = NF_MODE == "nfcore-gpu"

PARAMS_FILE = ROOT / "config/methylseq_params.yaml"
CONFIG_FILES = [ROOT / "config/methylseq.config"]
if GPU:
    CONFIG_FILES.append(ROOT / "config/methylseq_gpu.config")

# Nextflow state, inputs and work files (large) live under data/<mode>/
NF_DIR = ROOT / "data" / NF_MODE
SAMPLESHEET = NF_DIR / "inputs/samplesheet.csv"
TARGETS_UCSC = NF_DIR / "inputs/ELSA_plasma_2473_hg19.ucsc.bed"
WORK_DIR = NF_DIR / "work"

# Where 04_features.py looks in this mode
COV_DIR, COV_SUFFIX = COV_FILES[NF_MODE]

# nf-core/methylseq 4.2.0 includes conf/aws/batch/nextflow.config, a file
# missing from the release. Nextflow >= 25.10 parses configs strictly and
# fails on it; the legacy parser only reads the profiles in use.
NXF_ENV = {"NXF_SYNTAX_PARSER": "v1"}

# nf-core results (MultiQC, BAMs, methylation calls)
OUT_DIR = NFCORE_RESULTS[NF_MODE]

# bwa-meth index (nfcore-gpu): built by nf-core on the first run and saved
# to OUT_DIR/bwameth/reference_genome/; may also be put in reference/hg19/bwameth/
BWAMETH_INDEX_DIRS = [REF_DIR / "bwameth", OUT_DIR / "bwameth/reference_genome"]


def parse_args():

    p = argparse.ArgumentParser(
        description=f"Run nf-core/methylseq for the ELSA cohort ({NF_MODE} mode)."
    )
    p.add_argument("--profile", default="docker",
                   help="Nextflow profile: docker, singularity, apptainer, "
                        "conda, ... (default: %(default)s). nfcore-gpu mode "
                        "adds the gpu profile; conda cannot run on the GPU.")
    p.add_argument("--revision", default=REVISION,
                   help="Pipeline version (default: %(default)s).")
    p.add_argument("--build-index", action="store_true",
                   help="Let nf-core build the aligner index instead of using "
                        "reference/hg19/Bisulfite_Genome (Bismark) or a saved "
                        "bwa-meth index.")
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

    if not TARGET_BED.exists():
        sys.exit(
            f"Missing target file: {TARGET_BED}\n"
            "Run: python scripts/01_data.py"
        )

    TARGETS_UCSC.parent.mkdir(parents=True, exist_ok=True)

    with TARGET_BED.open() as src, TARGETS_UCSC.open("w") as dst:
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
            "Run: python scripts/02_reference.py"
        )

    profile = args.profile
    if GPU and "gpu" not in profile.split(","):
        profile += ",gpu"

    cmd = [
        "nextflow", "run", PIPELINE,
        "-r", args.revision,
        "-profile", profile,
        "-params-file", PARAMS_FILE,
        *(x for config in CONFIG_FILES for x in ("-c", config)),
        "-work-dir", WORK_DIR,
        "--input", SAMPLESHEET,
        "--outdir", OUT_DIR,
        "--fasta", FASTA,
        "--target_regions_file", TARGETS_UCSC,
    ]

    if GPU:
        cmd += ["--aligner", "bwameth"]
        index = None if args.build_index else bwameth_index()
        cmd += ["--bwameth_index", index] if index else ["--save_reference"]

    elif not args.build_index and (REF_DIR / "Bisulfite_Genome").exists():
        cmd += ["--bismark_index", REF_DIR]
    else:
        cmd += ["--save_reference"]

    if not args.no_resume:
        cmd.append("-resume")

    return [str(x) for x in cmd]


def bwameth_index():
    """Directory of a saved bwa-meth index (<genome>.bwameth.c2t.*), or None."""

    for folder in BWAMETH_INDEX_DIRS:
        found = sorted(folder.rglob("*.bwameth.c2t.bwt*")) if folder.exists() else []
        if found:
            return found[0].parent

    return None


# ============================================================
# Collect coverage files for 04_features.py
# ============================================================

def collect_coverage(run_ids):
    return collect_methyldackel(run_ids) if GPU else link_bismark_coverage(run_ids)


def report(done, run_ids, missing, ambiguous=()):

    print(f"\nCoverage files {done}: {len(run_ids) - len(missing) - len(ambiguous)}"
          f"/{len(run_ids)} -> {COV_DIR}")

    for run_id in missing:
        print(f"  MISSING   {run_id}")
    for run_id, matches in ambiguous:
        print(f"  AMBIGUOUS {run_id}: {', '.join(p.name for p in matches)}")

    return 1 if missing or ambiguous else 0


def bedgraph_to_cov(src, dst):
    """
    MethylDackel bedGraph (track line; chrom, start 0-based, end, percent,
    methylated, unmethylated) -> Bismark .cov.gz (chrom, start, end 1-based,
    percent, methylated, unmethylated). Written to a temporary file first.
    """

    tmp = dst.with_name(dst.name + ".tmp")

    with open(src) as fin, gzip.open(tmp, "wt") as fout:
        for line in fin:
            if line.startswith("track"):
                continue
            chrom, start0, _, percent, meth, unmeth = line.split()[:6]
            pos = int(start0) + 1
            fout.write(f"{chrom}\t{pos}\t{pos}\t{percent}\t{meth}\t{unmeth}\n")

    tmp.replace(dst)


def collect_methyldackel(run_ids):
    """
    Convert OUT_DIR/methyldackel/<run>.markdup.sorted_CpG.bedGraph to
    data/nfcore-gpu/methylation/<run>.bismark.cov.gz (skipped when up to date).
    """

    COV_DIR.mkdir(parents=True, exist_ok=True)
    missing = []

    for run_id in run_ids:
        src = OUT_DIR / "methyldackel" / f"{run_id}.markdup.sorted_CpG.bedGraph"
        dst = COV_DIR / f"{run_id}{COV_SUFFIX}"

        if not src.exists():
            missing.append(run_id)
            continue

        if not (dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime):
            bedgraph_to_cov(src, dst)

    return report("converted", run_ids, missing)


def link_bismark_coverage(run_ids):
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

    return report("linked", run_ids, missing, ambiguous)


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
            f'  sg docker -c "python main.py --mode {NF_MODE}"'
        )


def check_gpu(profile):
    """Fail early in nfcore-gpu mode without a usable NVIDIA GPU."""

    if "conda" in profile.split(",") or "mamba" in profile.split(","):
        sys.exit("The GPU module (Parabricks) needs a container profile: "
                 "docker, singularity or apptainer.")

    if shutil.which("nvidia-smi") is None or subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True
    ).returncode != 0:
        sys.exit(
            "No usable NVIDIA GPU: nvidia-smi is missing or fails.\n"
            "nfcore-gpu mode needs an NVIDIA GPU with its driver; use --mode nfcore otherwise."
        )

    if "docker" in profile.split(","):
        runtimes = subprocess.run(["docker", "info", "--format", "{{json .Runtimes}}"],
                                  capture_output=True, text=True).stdout
        if "nvidia" not in runtimes:
            sys.exit(
                "Docker has no nvidia runtime. Install the NVIDIA Container Toolkit:\n"
                "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html\n"
                "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
            )


def main():

    args = parse_args()
    print(f"Mode: {NF_MODE}")
    run_ids = runs_with_fastq()

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
        if GPU:
            check_gpu(args.profile)

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
