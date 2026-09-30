from pathlib import Path
import sys

from utils import ROOT, LOG_DIR, THREADS, run, all_exist, finish

FASTQ_DIR = ROOT / "data/fastq"
OUT_DIR = ROOT / "data/trimmed"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    failed = []

    for r1 in sorted(FASTQ_DIR.glob("*_1.fastq.gz")):

        r2 = Path(str(r1).replace("_1.fastq.gz", "_2.fastq.gz"))

        if not r2.exists():
            print(f"[SKIP] Missing R2: {r1.name}")
            continue

        run_id = r1.name.replace("_1.fastq.gz", "")

        # Trim Galore writes the validated pair last.
        if all_exist(
            OUT_DIR / f"{run_id}_1_val_1.fq.gz",
            OUT_DIR / f"{run_id}_2_val_2.fq.gz",
            OUT_DIR / f"{run_id}_2.fastq.gz_trimming_report.txt",
        ):
            print(f"[EXISTS] {run_id}")
            continue

        print(f"\n[TRIM] {run_id}")

        try:
            run(
                [
                    "trim_galore",
                    "--paired",
                    "--quality", "20",
                    "--illumina",
                    "--length", "20",
                    "--cores", THREADS,
                    "--output_dir", OUT_DIR,
                    r1,
                    r2,
                ],
                LOG_DIR / f"{run_id}.trim.log",
            )

        except RuntimeError as e:
            print(f"[FAILED] {run_id}: {e}")
            failed.append((run_id, e))

    return finish("Trim Galore", failed)


if __name__ == "__main__":
    sys.exit(main())
