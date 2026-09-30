"""
Run the whole pipeline, from metadata to the ML-ready dataset, in order.

Each step is run as its own process (python scripts/NN_*.py), exactly as
when it is run by hand. The steps are resumable, so rerunning main.py
skips the work that is already done.

Examples:
    python main.py                        # all steps
    python main.py --list                 # show the steps
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

# (id, script, description) in execution order
STEPS = [
    ("0",   "0_prepare_metadata.py",      "labels, manifests, target regions"),
    ("00",  "00_select_samples.py",       "balanced working cohort"),
    ("001", "001_download_fastq.py",      "download FASTQ from ENA"),
    ("01",  "01_fastqc.py",               "FastQC of raw reads"),
    ("02",  "02_trim.py",                 "adapter and quality trimming"),
    ("03",  "03_fastqc_trimmed.py",       "FastQC of trimmed reads"),
    ("04",  "04_prepare_references.py",   "hg19 reference and Bismark index"),
    ("05",  "05_align.py",                "bisulfite alignment"),
    ("06",  "06_dedup_filter.py",         "deduplication and read filtering"),
    ("07",  "07_extract_methylation.py",  "per-CpG methylation calls"),
    ("08",  "08_build_features.py",       "region-level methylation matrix"),
    ("09",  "09_feature_qc.py",           "feature-matrix QC"),
    ("10",  "10_prepare_ml_dataset.py",   "ML-ready dataset"),
]

STEP_IDS = [step_id for step_id, _, _ in STEPS]


def parse_args():

    p = argparse.ArgumentParser(
        description="Run the ctDNA methylation pipeline end to end."
    )
    p.add_argument("--from", dest="start", choices=STEP_IDS, default=STEP_IDS[0],
                   help="First step to run (default: %(default)s).")
    p.add_argument("--to", dest="end", choices=STEP_IDS, default=STEP_IDS[-1],
                   help="Last step to run (default: %(default)s).")
    p.add_argument("--skip", nargs="+", choices=STEP_IDS, default=[],
                   help="Steps to leave out.")
    p.add_argument("--threads", type=int,
                   help="Sets CTDNA_THREADS for all steps.")
    p.add_argument("--keep-going", action="store_true",
                   help="Continue with the next step when a step fails.")
    p.add_argument("--list", action="store_true",
                   help="List the steps and exit.")

    # Passed through to single steps
    p.add_argument("--strict", action="store_true",
                   help="Step 08: fail if any cohort sample is missing.")
    p.add_argument("--min-depth", type=int,
                   help="Step 10: minimum read depth per value.")
    p.add_argument("--max-missing", type=float,
                   help="Step 10: maximum missing fraction per region.")
    p.add_argument("--out-dir",
                   help="Step 10: output directory.")

    args = p.parse_args()

    if STEP_IDS.index(args.start) > STEP_IDS.index(args.end):
        p.error(f"--from {args.start} comes after --to {args.end}")

    return args


def step_args(step_id, args):
    """Extra command-line arguments for a single step."""
    extra = []

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

    if args.list:
        for step_id, script, description in STEPS:
            print(f"  {step_id:>4}  {script:<28} {description}")
        return 0

    first = STEP_IDS.index(args.start)
    last = STEP_IDS.index(args.end)
    selected = [s for s in STEPS[first:last + 1] if s[0] not in args.skip]

    env = os.environ.copy()
    if args.threads is not None:
        env["CTDNA_THREADS"] = str(args.threads)

    results = []
    pipeline_start = time.time()

    for n, (step_id, script, description) in enumerate(selected, start=1):

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
                  f"python main.py --from {step_id}")
            break

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Pipeline summary")
    print("=" * 70)

    for step_id, script, returncode, elapsed in results:
        status = "OK" if returncode == 0 else f"FAILED ({returncode})"
        print(f"  {step_id:>4}  {script:<28} {format_time(elapsed):>9}  {status}")

    not_run = selected[len(results):]
    for step_id, script, _ in not_run:
        print(f"  {step_id:>4}  {script:<28} {'':>9}  not run")

    print(f"\n  total time: {format_time(time.time() - pipeline_start)}")
    print("=" * 70)

    failed = any(returncode != 0 for _, _, returncode, _ in results)
    return 1 if failed or not_run else 0


if __name__ == "__main__":
    sys.exit(main())
