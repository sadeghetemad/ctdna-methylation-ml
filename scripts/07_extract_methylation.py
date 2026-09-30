import sys

from utils import ROOT, LOG_DIR, run, all_exist, finish

BAM_DIR = ROOT / "data/filtered"
REF_DIR = ROOT / "reference/hg19"
OUT_DIR = ROOT / "data/methylation"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    failed = []

    for bam in sorted(
        BAM_DIR.glob("*.filtered.namesort.bam")
    ):

        run_id = bam.name.replace(
            ".filtered.namesort.bam",
            "",
        )

        prefix = OUT_DIR / f"{run_id}.filtered.namesort"

        # The cytosine report and context summary are written last.
        if all_exist(
            f"{prefix}.bismark.cov.gz",
            f"{prefix}.CpG_report.txt.gz",
            f"{prefix}.cytosine_context_summary.txt",
        ):
            print(f"[EXISTS] {run_id}")
            continue

        print(f"\n[METHYLATION] {run_id}")

        try:
            run(
                [
                    "bismark",
                    "extract",
                    "--paired-end",
                    "--comprehensive",
                    "--gzip",
                    "--bedGraph",
                    "--cytosine_report",
                    "--genome_folder", REF_DIR,
                    "--output_dir", OUT_DIR,
                    bam,
                ],
                LOG_DIR / f"{run_id}.methylation.log",
            )

        except RuntimeError as e:
            print(f"[FAILED] {run_id}: {e}")
            failed.append((run_id, e))

    return finish("Methylation extraction", failed)


if __name__ == "__main__":
    sys.exit(main())
