"""
Shared paths and helpers for the numbered pipeline scripts.

Every script can be run on its own (python scripts/NN_*.py); Python puts the
script directory on sys.path, so `from utils import ...` works from anywhere.
"""

from pathlib import Path
import os
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = ROOT / "logs"

# Threads for FastQC / Trim Galore / Bismark / samtools.
# Override without editing code:  CTDNA_THREADS=16 python scripts/05_align.py
THREADS = int(os.environ.get("CTDNA_THREADS", "8"))


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
