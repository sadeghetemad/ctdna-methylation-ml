from pathlib import Path
import sys

from utils import ROOT, LOG_DIR, THREADS, run, all_exist, finish

FASTQ_DIR = ROOT / "data/fastq"
OUT_DIR = ROOT / "results/fastqc_raw"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def fastqc_html(fastq):
    # SRR123_1.fastq.gz -> SRR123_1_fastqc.html
    name = fastq.name
    for ext in (".fastq.gz", ".fq.gz", ".fastq", ".fq"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return OUT_DIR / f"{name}_fastqc.html"


def main():

    r1_files = sorted(FASTQ_DIR.glob("*_1.fastq.gz"))
    failed = []

    for r1 in r1_files:

        r2 = Path(str(r1).replace("_1.fastq.gz", "_2.fastq.gz"))

        if not r2.exists():
            print(f"[SKIP] Missing R2: {r1.name}")
            continue

        run_id = r1.name.replace("_1.fastq.gz", "")

        if all_exist(fastqc_html(r1), fastqc_html(r2)):
            print(f"[EXISTS] {run_id}")
            continue

        print(f"\n[FASTQC] {run_id}")

        try:
            run(
                [
                    "fastqc",
                    "-t", THREADS,
                    "-o", OUT_DIR,
                    r1,
                    r2,
                ],
                LOG_DIR / f"{run_id}.fastqc_raw.log",
            )

        except RuntimeError as e:
            print(f"[FAILED] {run_id}: {e}")
            failed.append((run_id, e))

    return finish("FastQC (raw)", failed)


if __name__ == "__main__":
    sys.exit(main())
