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

AWS Batch (nfcore mode): every sample runs as its own Spot job, with the
FASTQ files, reference and Nextflow work files in S3. Stage the inputs with
scripts/03_aws_stage.py first; see README, "Running on AWS Batch".
    python scripts/03_process_nfcore.py --aws-batch
    python scripts/03_process_nfcore.py --aws-batch --collect-only

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
    BAM_FILES, fastq_pair, runs_with_fastq, cohort_runs,
)


PIPELINE = "nf-core/methylseq"
REVISION = "4.2.0"

NF_MODE = MODE if MODE in NFCORE_RESULTS else "nfcore"
GPU = NF_MODE == "nfcore-gpu"

PARAMS_FILE = ROOT / "config/methylseq_params.yaml"
CONFIG_FILES = [ROOT / "config/methylseq.config"]
if GPU:
    CONFIG_FILES.append(ROOT / "config/methylseq_gpu.config")

# AWS Batch (--aws-batch): the config comes last so it overrides the
# local resource limits of methylseq.config
AWSBATCH_CONFIG = ROOT / "config/methylseq_awsbatch.config"

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
    p.add_argument("--aws-batch", action="store_true",
                   help="Run the samples as AWS Batch jobs, with inputs, work "
                        "files and results in s3://$CTDNA_S3_BUCKET (nfcore mode "
                        "only). Stage the inputs first with scripts/03_aws_stage.py.")
    p.add_argument("--no-bams", action="store_true",
                   help="With --aws-batch: do not download the deduplicated BAMs "
                        "(needed by step 05) from S3.")
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

def write_samplesheet(run_ids, s3=None):

    rows = []

    for run_id in run_ids:
        if s3:
            r1, r2 = (f"{s3}/fastq/{run_id}_{n}.fastq.gz" for n in (1, 2))
        else:
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

def nextflow_command(args, s3=None):

    if not FASTA.exists() and not s3:
        sys.exit(
            f"Missing reference: {FASTA}\n"
            "Run: python scripts/02_reference.py"
        )

    profile = args.profile
    if GPU and "gpu" not in profile.split(","):
        profile += ",gpu"

    if s3:
        work_dir, outdir = f"{s3}/work", f"{s3}/results/{OUT_DIR.name}"
        fasta, targets = f"{s3}/reference/hg19/hg19.fa", f"{s3}/inputs/{TARGETS_UCSC.name}"
        configs = [*CONFIG_FILES, AWSBATCH_CONFIG]
    else:
        work_dir, outdir, fasta, targets = WORK_DIR, OUT_DIR, FASTA, TARGETS_UCSC
        configs = CONFIG_FILES

    cmd = [
        "nextflow", "run", PIPELINE,
        "-r", args.revision,
        "-profile", profile,
        "-params-file", PARAMS_FILE,
        *(x for config in configs for x in ("-c", config)),
        "-work-dir", work_dir,
        "--input", SAMPLESHEET,
        "--outdir", outdir,
        "--fasta", fasta,
        "--target_regions_file", targets,
    ]

    if GPU:
        cmd += ["--aligner", "bwameth"]
        index = None if args.build_index else bwameth_index()
        cmd += ["--bwameth_index", index] if index else ["--save_reference"]

    elif not args.build_index and s3:
        cmd += ["--bismark_index", f"{s3}/reference/hg19"]
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


# ============================================================
# AWS Batch (--aws-batch)
# ============================================================

def s3_base():
    bucket = os.environ.get("CTDNA_S3_BUCKET")
    missing = [v for v in ("CTDNA_S3_BUCKET", "CTDNA_BATCH_QUEUE", "CTDNA_BATCH_JOB_ROLE", "AWS_REGION")
               if not os.environ.get(v)]

    if missing:
        sys.exit(
            f"Not set: {', '.join(missing)}\n"
            "Run: source <(terraform -chdir=infra/aws output -raw env_exports)"
        )

    if NF_MODE != "nfcore":
        sys.exit("--aws-batch supports nfcore mode only (the GPU mode needs GPU instances).")

    return f"s3://{bucket}"


def s3_runs_with_fastq(s3):
    """Cohort runs with both FASTQ files in S3 (uploaded by 03_aws_stage.py)."""

    listing = subprocess.run(
        ["aws", "s3", "ls", f"{s3}/fastq/"], capture_output=True, text=True
    )

    if listing.returncode != 0:
        sys.exit(f"Cannot list {s3}/fastq/:\n{listing.stderr.strip()}")

    present = {line.split()[-1] for line in listing.stdout.splitlines() if line.strip()}
    run_ids = cohort_runs()
    ready = [r for r in run_ids if {f"{r}_1.fastq.gz", f"{r}_2.fastq.gz"} <= present]
    missing = [r for r in run_ids if r not in ready]

    if missing:
        print(f"WARNING: FASTQ files missing in S3 for {len(missing)} run(s), left out: "
              f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
        print("         Run: python scripts/03_aws_stage.py fastq")

    if not ready:
        sys.exit(f"No run of the cohort has both FASTQ files in {s3}/fastq/")

    return ready


def upload_inputs(s3):
    """The samplesheet and targets files are small; the jobs read them from S3."""

    for path in (SAMPLESHEET, TARGETS_UCSC):
        subprocess.run(["aws", "s3", "cp", path, f"{s3}/inputs/{path.name}", "--only-show-errors"],
                       check=True)


def download_results(s3, bams):
    """
    Download the files the later steps read from s3://.../results/<name>/ to
    OUT_DIR: coverage files (step 04), the deduplicated BAMs with their
    indexes (step 05, optional) and the MultiQC report.
    """

    includes = ["*.bismark.cov.gz", "multiqc/*", "pipeline_info/*"]
    bam_dir, _ = BAM_FILES[NF_MODE]

    if bams:
        includes.append(f"{bam_dir.relative_to(OUT_DIR)}/*")

    cmd = ["aws", "s3", "sync", f"{s3}/results/{OUT_DIR.name}", str(OUT_DIR),
           "--only-show-errors", "--exclude", "*"]
    cmd += [x for pattern in includes for x in ("--include", pattern)]

    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main():

    args = parse_args()
    print(f"Mode: {NF_MODE}")

    s3 = s3_base() if args.aws_batch else None
    run_ids = s3_runs_with_fastq(s3) if s3 else runs_with_fastq()

    if not args.collect_only:

        write_samplesheet(run_ids, s3)
        write_ucsc_targets()
        cmd = nextflow_command(args, s3)

        print()
        print("$ " + " ".join(f"{k}={v}" for k, v in NXF_ENV.items()) + " " + shlex.join(cmd))

        if args.dry_run:
            return 0

        if shutil.which("nextflow") is None:
            sys.exit(
                "nextflow not found. Install it into the environment:\n"
                "  conda env update -f environment.yml"
            )

        if s3:
            upload_inputs(s3)
        else:
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

    if s3:
        returncode = download_results(s3, bams=not args.no_bams)
        if returncode != 0:
            return returncode

    return collect_coverage(run_ids)


if __name__ == "__main__":
    sys.exit(main())
