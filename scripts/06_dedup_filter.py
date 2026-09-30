import sys

from utils import (
    ROOT, LOG_DIR, THREADS,
    run, run_pipe, all_exist, tmp_path, finish,
)

ALIGN_DIR = ROOT / "data/aligned"
DEDUP_DIR = ROOT / "data/dedup"
FILTER_DIR = ROOT / "data/filtered"

DEDUP_DIR.mkdir(parents=True, exist_ok=True)
FILTER_DIR.mkdir(parents=True, exist_ok=True)

MIN_MAPQ = 20


def main():

    failed = []

    for bam in sorted(ALIGN_DIR.glob("*_pe.bam")):

        run_id = bam.name.replace("_pe.bam", "")
        log = LOG_DIR / f"{run_id}.dedup_filter.log"

        print(f"\n[PROCESS] {run_id}")

        if not (ALIGN_DIR / f"{run_id}_PE_report.txt").exists():
            print("[SKIP] alignment not finished (no PE report)")
            continue

        dedup_bam = DEDUP_DIR / f"{run_id}_pe.deduplicated.bam"
        dedup_report = DEDUP_DIR / f"{run_id}_pe.deduplication_report.txt"

        filtered_bam = FILTER_DIR / f"{run_id}.filtered.bam"
        filtered_bai = FILTER_DIR / f"{run_id}.filtered.bam.bai"

        namesort_bam = FILTER_DIR / f"{run_id}.filtered.namesort.bam"

        try:

            # ------------------------------------------------
            # Deduplication
            # ------------------------------------------------

            if all_exist(dedup_bam, dedup_report):
                print("[EXISTS] deduplicated BAM")

            else:
                print("[DEDUP]")

                run(
                    [
                        "bismark",
                        "dedup",
                        "--paired",
                        "--bam",
                        "--parallel", "4",
                        "--output_dir", DEDUP_DIR,
                        bam,
                    ],
                    log,
                )

            # ------------------------------------------------
            # MAPQ / proper-pair filter + coordinate sort
            #
            # -q 20     : MAPQ >= 20
            # -f 2      : proper pair
            # -F 0x904  : drop unmapped, secondary, supplementary
            #
            # Written to a temporary file and renamed only after
            # success, so a crash never leaves a "complete-looking"
            # truncated BAM behind.
            # ------------------------------------------------

            if all_exist(filtered_bam, filtered_bai):
                print("[EXISTS] filtered BAM")

            else:
                print(f"[FILTER MAPQ>={MIN_MAPQ}]")

                tmp = tmp_path(filtered_bam)

                run_pipe(
                    [
                        "samtools", "view",
                        "-@", THREADS,
                        "-b",
                        "-q", MIN_MAPQ,
                        "-f", "2",
                        "-F", "0x904",
                        dedup_bam,
                    ],
                    [
                        "samtools", "sort",
                        "-@", THREADS,
                        "-O", "bam",
                        "-o", tmp,
                        "-",
                    ],
                    log,
                )

                tmp.replace(filtered_bam)

                run(
                    ["samtools", "index", "-@", THREADS, filtered_bam],
                    log,
                )

                # Name-sorted BAM derives from the filtered BAM,
                # so it must be rebuilt as well.
                namesort_bam.unlink(missing_ok=True)

            # ------------------------------------------------
            # Name sort (required by the methylation extractor)
            # ------------------------------------------------

            if namesort_bam.exists():
                print("[EXISTS] name-sorted BAM")

            else:
                print("[NAME SORT]")

                tmp = tmp_path(namesort_bam)

                run(
                    [
                        "samtools", "sort",
                        "-@", THREADS,
                        "-n",
                        "-O", "bam",
                        "-o", tmp,
                        filtered_bam,
                    ],
                    log,
                )

                tmp.replace(namesort_bam)

        except RuntimeError as e:
            print(f"[FAILED] {run_id}: {e}")
            failed.append((run_id, e))

    return finish("Dedup / filter", failed)


if __name__ == "__main__":
    sys.exit(main())
