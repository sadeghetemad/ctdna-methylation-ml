"""
Run the whole pipeline, from metadata to the ML-ready dataset, in order.

Two modes for the read processing (FASTQ -> per-CpG methylation calls):
    classic : the Python scripts 01-07 (FastQC, Trim Galore, Bismark, samtools)
    nfcore  : nf-core/methylseq (step "nf") in place of 01-07

Each step is run as its own process (python scripts/NN_*.py), exactly as
when it is run by hand. The steps are resumable, so rerunning main.py
skips the work that is already done.

Examples:
    python main.py                        # classic mode, all steps
    python main.py --mode nfcore          # nf-core/methylseq for 01-07
    python main.py --mode nfcore --list   # show the steps of a mode
    python main.py --from 05              # start at alignment
    python main.py --from 08 --to 09      # only features and feature QC
    python main.py --skip 01 03           # without FastQC
    python main.py --threads 16 --strict
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

MODES = ("classic", "nfcore")
BOTH = MODES

# (id, script, description, modes) in execution order
STEPS = [
    ("0",   "0_prepare_metadata.py",      "labels, manifests, target regions",  BOTH),
    ("00",  "00_select_samples.py",       "balanced working cohort",            BOTH),
    ("001", "001_download_fastq.py",      "download FASTQ from ENA",            BOTH),
    ("01",  "01_fastqc.py",               "FastQC of raw reads",                ("classic",)),
    ("02",  "02_trim.py",                 "adapter and quality trimming",       ("classic",)),
    ("03",  "03_fastqc_trimmed.py",       "FastQC of trimmed reads",            ("classic",)),
    ("04",  "04_prepare_references.py",   "hg19 reference and Bismark index",   BOTH),
    ("nf",  "nf_methylseq.py",            "nf-core/methylseq: QC to methylation calls", ("nfcore",)),
    ("05",  "05_align.py",                "bisulfite alignment",                ("classic",)),
    ("06",  "06_dedup_filter.py",         "deduplication and read filtering",   ("classic",)),
    ("07",  "07_extract_methylation.py",  "per-CpG methylation calls",          ("classic",)),
    ("08",  "08_build_features.py",       "region-level methylation matrix",    BOTH),
    ("09",  "09_feature_qc.py",           "feature-matrix QC",                  BOTH),
    ("10",  "10_prepare_ml_dataset.py",   "ML-ready dataset",                   BOTH),
]

STEP_IDS = [step[0] for step in STEPS]


def mode_steps(mode):
    return [step for step in STEPS if mode in step[3]]


def parse_args():

    p = argparse.ArgumentParser(
        description="Run the ctDNA methylation pipeline end to end."
    )
    p.add_argument("--mode", choices=MODES, default="classic",
                   help="How steps 01-07 are run (default: %(default)s).")
    p.add_argument("--from", dest="start", choices=STEP_IDS,
                   help="First step to run (default: the first step).")
    p.add_argument("--to", dest="end", choices=STEP_IDS,
                   help="Last step to run (default: the last step).")
    p.add_argument("--skip", nargs="+", choices=STEP_IDS, default=[],
                   help="Steps to leave out.")
    p.add_argument("--threads", type=int,
                   help="Sets CTDNA_THREADS for all steps.")
    p.add_argument("--keep-going", action="store_true",
                   help="Continue with the next step when a step fails.")
    p.add_argument("--list", action="store_true",
                   help="List the steps of the mode and exit.")

    # Passed through to single steps
    p.add_argument("--nf-profile", default="docker",
                   help="Step nf: Nextflow profile (default: %(default)s).")
    p.add_argument("--build-index", action="store_true",
                   help="Step nf: let nf-core build its own Bismark index.")
    p.add_argument("--strict", action="store_true",
                   help="Step 08: fail if any cohort sample is missing.")
    p.add_argument("--min-depth", type=int,
                   help="Step 10: minimum read depth per value.")
    p.add_argument("--max-missing", type=float,
                   help="Step 10: maximum missing fraction per region.")
    p.add_argument("--out-dir",
                   help="Step 10: output directory.")

    args = p.parse_args()

    ids = [step[0] for step in mode_steps(args.mode)]

    for option, step_id in (("--from", args.start), ("--to", args.end)):
        if step_id is not None and step_id not in ids:
            p.error(f"{option} {step_id}: not a step of {args.mode} mode "
                    f"({', '.join(ids)})")

    args.start = args.start or ids[0]
    args.end = args.end or ids[-1]

    if ids.index(args.start) > ids.index(args.end):
        p.error(f"--from {args.start} comes after --to {args.end}")

    return args


def step_args(step_id, args):
    """Extra command-line arguments for a single step."""
    extra = []

    if step_id == "nf":
        extra += ["--profile", args.nf_profile]
        if args.build_index:
            extra.append("--build-index")

    if step_id == "08" and args.strict:
        extra.append("--strict")

    if step_id == "10":
        if args.min_depth is not None:
            extra += ["--min-depth", str(args.min_depth)]
        if args.max_missing is not None:
            extra += ["--max-missing", str(args.max_missing)]
        if args.out_dir is not None:
            extra += ["--out-dir", args.out_dir]

    return extra


def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def main():

    args = parse_args()
    steps = mode_steps(args.mode)

    if args.list:
        print(f"Steps in {args.mode} mode:")
        for step_id, script, description, _ in steps:
            print(f"  {step_id:>4}  {script:<28} {description}")
        return 0

    ids = [step[0] for step in steps]
    first = ids.index(args.start)
    last = ids.index(args.end)
    selected = [s for s in steps[first:last + 1] if s[0] not in args.skip]

    # Every step reads the mode from utils.MODE
    env = os.environ.copy()
    env["CTDNA_MODE"] = args.mode
    if args.threads is not None:
        env["CTDNA_THREADS"] = str(args.threads)

    print(f"Mode: {args.mode}")

    results = []
    pipeline_start = time.time()

    for n, (step_id, script, description, _) in enumerate(selected, start=1):

        cmd = [sys.executable, str(SCRIPTS / script)] + step_args(step_id, args)

        print()
        print("#" * 70)
        print(f"# [{n}/{len(selected)}] step {step_id}: {description}")
        print(f"# $ {' '.join(cmd[1:])}")
        print("#" * 70, flush=True)

        start = time.time()
        returncode = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        elapsed = time.time() - start

        results.append((step_id, script, returncode, elapsed))

        if returncode != 0 and not args.keep_going:
            print(f"\nStep {step_id} failed (exit code {returncode}); stopping.")
            print("Fix the problem and resume with:  "
                  f"python main.py --mode {args.mode} --from {step_id}")
            break

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(f"Pipeline summary ({args.mode} mode)")
    print("=" * 70)

    for step_id, script, returncode, elapsed in results:
        status = "OK" if returncode == 0 else f"FAILED ({returncode})"
        print(f"  {step_id:>4}  {script:<28} {format_time(elapsed):>9}  {status}")

    not_run = selected[len(results):]
    for step_id, script, _, _ in not_run:
        print(f"  {step_id:>4}  {script:<28} {'':>9}  not run")

    print(f"\n  total time: {format_time(time.time() - pipeline_start)}")
    print("=" * 70)

    failed = any(returncode != 0 for _, _, returncode, _ in results)
    return 1 if failed or not_run else 0


if __name__ == "__main__":
    sys.exit(main())
