"""
Run the whole pipeline, from metadata to the ML-ready dataset, in order.

    01 data        metadata, working cohort, FASTQ download
    02 reference   hg19 and its Bismark index
    03 process     FASTQ -> per-CpG methylation calls, in one of three modes:
                     custom     : our own commands (03_process_custom.py)
                     nfcore     : nf-core/methylseq, Bismark (03_process_nfcore.py)
                     nfcore-gpu : nf-core/methylseq, bwa-meth on an NVIDIA GPU
    04 features    region-level methylation matrix
    05 reads       read-level methylation and fragment features
    06 qc          feature-matrix QC
    07 ml          ML-ready dataset

Each step is run as its own process (python scripts/NN_*.py), exactly as
when it is run by hand. The steps are resumable, so rerunning main.py
skips the work that is already done.

Examples:
    python main.py                        # custom mode, all steps
    python main.py --mode nfcore          # nf-core/methylseq for step 03
    python main.py --mode nfcore-gpu      # the same with GPU alignment
    python main.py --list                 # show the steps of a mode
    python main.py --from 03              # resume at the read processing
    python main.py --from 04 --to 06      # only features and feature QC
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

MODES = ("custom", "nfcore", "nfcore-gpu")

# (id, description, script; {mode: script} where the modes differ)
STEPS = [
    ("01", "metadata, cohort, FASTQ download",   "01_data.py"),
    ("02", "hg19 reference and Bismark index",   "02_reference.py"),
    ("03", "FASTQ -> per-CpG methylation calls", {"custom": "03_process_custom.py",
                                                  "nfcore": "03_process_nfcore.py",
                                                  "nfcore-gpu": "03_process_nfcore.py"}),
    ("04", "region-level methylation matrix",    "04_features.py"),
    ("05", "read-level and fragment features",   "05_read_features.py"),
    ("06", "feature-matrix QC",                  "06_feature_qc.py"),
    ("07", "ML-ready dataset",                   "07_ml_dataset.py"),
]

STEP_IDS = [step[0] for step in STEPS]


def mode_steps(mode):
    """(id, description, script) of the steps in this mode."""
    return [
        (step_id, description, script[mode] if isinstance(script, dict) else script)
        for step_id, description, script in STEPS
    ]


def parse_args():

    p = argparse.ArgumentParser(
        description="Run the ctDNA methylation pipeline end to end."
    )
    p.add_argument("--mode", choices=MODES, default="custom",
                   help="How step 03 is run (default: %(default)s).")
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
    p.add_argument("--strict", action="store_true",
                   help="Steps 01, 04 and 05: fail if any cohort run is missing "
                        "(default: warn and continue without it).")
    p.add_argument("--nf-profile", default="docker",
                   help="Step 03 (nf-core modes): Nextflow profile (default: %(default)s); "
                        "nfcore-gpu adds the gpu profile.")
    p.add_argument("--build-index", action="store_true",
                   help="Step 03 (nf-core modes): let nf-core build its own aligner index.")
    p.add_argument("--min-depth", type=int,
                   help="Step 07: minimum read depth per value.")
    p.add_argument("--max-missing", type=float,
                   help="Step 07: maximum missing fraction per region.")
    p.add_argument("--out-dir",
                   help="Step 07: output directory.")

    args = p.parse_args()

    args.start = args.start or STEP_IDS[0]
    args.end = args.end or STEP_IDS[-1]

    if STEP_IDS.index(args.start) > STEP_IDS.index(args.end):
        p.error(f"--from {args.start} comes after --to {args.end}")

    return args


def step_args(step_id, args):
    """Extra command-line arguments for a single step."""
    extra = []

    if step_id in ("01", "04", "05") and args.strict:
        extra.append("--strict")

    if step_id == "03" and args.mode != "custom":
        extra += ["--profile", args.nf_profile]
        if args.build_index:
            extra.append("--build-index")

    if step_id == "07":
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
        for step_id, description, script in steps:
            print(f"  {step_id}  {script:<22} {description}")
        return 0

    first = STEP_IDS.index(args.start)
    last = STEP_IDS.index(args.end)
    selected = [s for s in steps[first:last + 1] if s[0] not in args.skip]

    # Every step reads the mode from utils.MODE
    env = os.environ.copy()
    env["CTDNA_MODE"] = args.mode
    if args.threads is not None:
        env["CTDNA_THREADS"] = str(args.threads)

    print(f"Mode: {args.mode}")

    results = []
    pipeline_start = time.time()

    for n, (step_id, description, script) in enumerate(selected, start=1):

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
        print(f"  {step_id}  {script:<22} {format_time(elapsed):>9}  {status}")

    not_run = selected[len(results):]
    for step_id, _, script in not_run:
        print(f"  {step_id}  {script:<22} {'':>9}  not run")

    print(f"\n  total time: {format_time(time.time() - pipeline_start)}")
    print("=" * 70)

    failed = any(returncode != 0 for _, _, returncode, _ in results)
    return 1 if failed or not_run else 0


if __name__ == "__main__":
    sys.exit(main())
