import sys

from utils import ROOT, LOG_DIR, THREADS, run, all_exist, finish

TRIM_DIR = ROOT / "data/trimmed"
OUT_DIR = ROOT / "results/fastqc_trimmed"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    r1_files = sorted(TRIM_DIR.glob("*_1_val_1.fq.gz"))
    failed = []

    print(f"Trimmed R1 files: {len(r1_files)}")

    for r1 in r1_files:

        run_id = r1.name.split("_1_val_1")[0]
        r2 = TRIM_DIR / f"{run_id}_2_val_2.fq.gz"

        if not r2.exists():
            print(f"[SKIP] Missing trimmed R2: {run_id}")
            continue

        if all_exist(
            OUT_DIR / f"{run_id}_1_val_1_fastqc.html",
            OUT_DIR / f"{run_id}_2_val_2_fastqc.html",
        ):
            print(f"[EXISTS] {run_id}")
            continue

        print(f"\n[FASTQC TRIMMED] {run_id}")

        try:
            run(
                [
                    "fastqc",
                    "-t", THREADS,
                    "-o", OUT_DIR,
                    r1,
                    r2,
                ],
                LOG_DIR / f"{run_id}.fastqc_trimmed.log",
            )

        except RuntimeError as e:
            print(f"[FAILED] {run_id}: {e}")
            failed.append((run_id, e))

    return finish("FastQC (trimmed)", failed)


if __name__ == "__main__":
    sys.exit(main())
