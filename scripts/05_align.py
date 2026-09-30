import sys

from utils import ROOT, LOG_DIR, THREADS, run, all_exist, finish

TRIM_DIR = ROOT / "data/trimmed"
REF_DIR = ROOT / "reference/hg19"
OUT_DIR = ROOT / "data/aligned"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    r1_files = sorted(TRIM_DIR.glob("*_1_val_1.fq.gz"))
    failed = []

    for r1 in r1_files:

        run_id = r1.name.split("_1_val_1")[0]
        r2 = TRIM_DIR / f"{run_id}_2_val_2.fq.gz"

        if not r2.exists():
            print(f"[SKIP] Missing trimmed R2: {run_id}")
            continue

        expected_bam = OUT_DIR / f"{run_id}_pe.bam"
        report = OUT_DIR / f"{run_id}_PE_report.txt"

        # The BAM is written progressively; the report only appears
        # when alignment finished. A BAM without a report is partial.
        if all_exist(expected_bam, report):
            print(f"[EXISTS] {expected_bam.name}")
            continue

        print(f"\n[ALIGN] {run_id}")

        try:
            run(
                [
                    "bismark",
                    "align",
                    "--genome", REF_DIR,
                    "-1", r1,
                    "-2", r2,
                    "--bowtie2",
                    "-p", THREADS,
                    "--output_dir", OUT_DIR,
                    "--basename", run_id,
                ],
                LOG_DIR / f"{run_id}.align.log",
            )

        except RuntimeError as e:
            print(f"[FAILED] {run_id}: {e}")
            failed.append((run_id, e))

    return finish("Bismark alignment", failed)


if __name__ == "__main__":
    sys.exit(main())
