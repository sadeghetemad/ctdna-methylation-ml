"""
Step 03, custom mode: FASTQ -> per-CpG methylation calls with our own
commands. nfcore mode runs 03_process_nfcore.py instead.

Each cohort run goes through the stages in order:

    fastqc          FastQC of the raw reads           results/fastqc_raw/
    trim            Trim Galore                       data/trimmed/
    fastqc_trimmed  FastQC of the trimmed reads       results/fastqc_trimmed/
    align           Bismark + Bowtie2                 data/aligned/
    dedup           bismark dedup                     data/dedup/
    filter          MAPQ / proper-pair filter, sort   data/filtered/
    extract         bismark extract                   data/methylation/

A stage is skipped when its final outputs exist. When a stage fails, the
rest of that run is skipped and the next run starts.

Examples:
    python scripts/03_process_custom.py
    python scripts/03_process_custom.py --stages dedup filter extract
    python scripts/03_process_custom.py --runs SRR9982549
"""

import argparse
import sys

from utils import (
    ROOT, LOG_DIR, THREADS, REF_DIR, CUSTOM_COV,
    fastq_pair, runs_with_fastq, run, run_pipe, all_exist, tmp_path, finish,
)


FASTQC_RAW_DIR = ROOT / "results/fastqc_raw"
FASTQC_TRIMMED_DIR = ROOT / "results/fastqc_trimmed"
TRIM_DIR = ROOT / "data/trimmed"
ALIGN_DIR = ROOT / "data/aligned"
DEDUP_DIR = ROOT / "data/dedup"
FILTER_DIR = ROOT / "data/filtered"
METH_DIR = CUSTOM_COV[0]    # read by 04_features.py

MIN_MAPQ = 20


# ============================================================
# File names
# ============================================================

def trimmed_pair(run_id):
    return TRIM_DIR / f"{run_id}_1_val_1.fq.gz", TRIM_DIR / f"{run_id}_2_val_2.fq.gz"


def aligned_bam(run_id):
    return ALIGN_DIR / f"{run_id}_pe.bam"


def dedup_bam(run_id):
    return DEDUP_DIR / f"{run_id}_pe.deduplicated.bam"


def filtered_bam(run_id):
    return FILTER_DIR / f"{run_id}.filtered.bam"


def namesort_bam(run_id):
    return FILTER_DIR / f"{run_id}.filtered.namesort.bam"


def log(run_id, stage):
    return LOG_DIR / f"{run_id}.{stage}.log"


def require(*paths):
    """Inputs of a stage; missing ones mean an earlier stage has not run."""
    missing = [p.relative_to(ROOT) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(f"missing input {', '.join(map(str, missing))}; run the earlier stages")


def exists(*paths):
    if all_exist(*paths):
        print("    [EXISTS]")
        return True
    return False


# ============================================================
# Stages: each one skips itself when its outputs exist
# ============================================================

def fastqc(run_id, reads, out_dir, stage):

    require(*reads)

    # SRR1_1.fastq.gz -> SRR1_1_fastqc.html, SRR1_1_val_1.fq.gz -> SRR1_1_val_1_fastqc.html
    stems = [r.name.removesuffix(".gz").rsplit(".", 1)[0] for r in reads]
    if exists(*(out_dir / f"{s}_fastqc.html" for s in stems)):
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    run(["fastqc", "-t", THREADS, "-o", out_dir, *reads], log(run_id, stage))


def fastqc_raw(run_id):
    fastqc(run_id, fastq_pair(run_id), FASTQC_RAW_DIR, "fastqc_raw")


def fastqc_trimmed(run_id):
    fastqc(run_id, trimmed_pair(run_id), FASTQC_TRIMMED_DIR, "fastqc_trimmed")


def trim(run_id):

    r1, r2 = fastq_pair(run_id)

    # Trim Galore writes the validated pair last
    if exists(*trimmed_pair(run_id), TRIM_DIR / f"{run_id}_2.fastq.gz_trimming_report.txt"):
        return

    TRIM_DIR.mkdir(parents=True, exist_ok=True)
    run(
        ["trim_galore", "--paired", "--quality", "20", "--illumina", "--length", "20",
         "--cores", THREADS, "--output_dir", TRIM_DIR, r1, r2],
        log(run_id, "trim"),
    )


def align(run_id):

    r1, r2 = trimmed_pair(run_id)
    require(r1, r2)

    # The BAM is written progressively; the report only appears when
    # the alignment finished. A BAM without a report is partial.
    if exists(aligned_bam(run_id), ALIGN_DIR / f"{run_id}_PE_report.txt"):
        return

    ALIGN_DIR.mkdir(parents=True, exist_ok=True)
    run(
        ["bismark", "align", "--genome", REF_DIR, "-1", r1, "-2", r2, "--bowtie2",
         "-p", THREADS, "--output_dir", ALIGN_DIR, "--basename", run_id],
        log(run_id, "align"),
    )


def dedup(run_id):

    require(aligned_bam(run_id), ALIGN_DIR / f"{run_id}_PE_report.txt")

    if exists(dedup_bam(run_id), DEDUP_DIR / f"{run_id}_pe.deduplication_report.txt"):
        return

    DEDUP_DIR.mkdir(parents=True, exist_ok=True)
    run(
        ["bismark", "dedup", "--paired", "--bam", "--parallel", "4",
         "--output_dir", DEDUP_DIR, aligned_bam(run_id)],
        log(run_id, "dedup_filter"),
    )


def filter_reads(run_id):
    """
    Coordinate-sorted, indexed BAM of the reads kept by
        -q 20     MAPQ >= 20
        -f 2      proper pair
        -F 0x904  no unmapped, secondary or supplementary alignments
    plus a name-sorted copy for bismark extract. Both are written to a
    temporary file and renamed only on success.
    """

    require(dedup_bam(run_id))

    bam, namesort = filtered_bam(run_id), namesort_bam(run_id)
    bai = bam.with_name(bam.name + ".bai")

    if exists(bam, bai, namesort):
        return

    FILTER_DIR.mkdir(parents=True, exist_ok=True)
    log_file = log(run_id, "dedup_filter")

    if not all_exist(bam, bai):
        tmp = tmp_path(bam)
        run_pipe(
            ["samtools", "view", "-@", THREADS, "-b", "-q", MIN_MAPQ, "-f", "2",
             "-F", "0x904", dedup_bam(run_id)],
            ["samtools", "sort", "-@", THREADS, "-O", "bam", "-o", tmp, "-"],
            log_file,
        )
        tmp.replace(bam)
        run(["samtools", "index", "-@", THREADS, bam], log_file)

        # Derived from the filtered BAM, so it is rebuilt as well
        namesort.unlink(missing_ok=True)

    tmp = tmp_path(namesort)
    run(["samtools", "sort", "-@", THREADS, "-n", "-O", "bam", "-o", tmp, bam], log_file)
    tmp.replace(namesort)


def extract(run_id):

    require(namesort_bam(run_id))

    # The cytosine report and context summary are written last
    prefix = f"{run_id}.filtered.namesort"
    if exists(*(METH_DIR / f"{prefix}{s}" for s in (
        ".bismark.cov.gz", ".CpG_report.txt.gz", ".cytosine_context_summary.txt",
    ))):
        return

    METH_DIR.mkdir(parents=True, exist_ok=True)
    run(
        ["bismark", "extract", "--paired-end", "--comprehensive", "--gzip", "--bedGraph",
         "--cytosine_report", "--genome_folder", REF_DIR, "--output_dir", METH_DIR,
         namesort_bam(run_id)],
        log(run_id, "methylation"),
    )


STAGES = {
    "fastqc": fastqc_raw,
    "trim": trim,
    "fastqc_trimmed": fastqc_trimmed,
    "align": align,
    "dedup": dedup,
    "filter": filter_reads,
    "extract": extract,
}


# ============================================================
# Main
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description="custom mode: FASTQ -> per-CpG methylation calls."
    )
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES),
                   help="Stages to run, always in pipeline order (default: all).")
    p.add_argument("--runs", nargs="+",
                   help="Only these cohort runs (default: all with FASTQ files).")
    return p.parse_args()


def main():

    args = parse_args()

    run_ids = runs_with_fastq()
    if args.runs:
        unknown = set(args.runs) - set(run_ids)
        if unknown:
            sys.exit(f"Not cohort runs with FASTQ files: {', '.join(sorted(unknown))}")
        run_ids = [r for r in run_ids if r in args.runs]

    stages = [s for s in STAGES if s in args.stages]
    failed = []

    for n, run_id in enumerate(run_ids, start=1):

        print()
        print(f"[{n}/{len(run_ids)}] {run_id}")

        for stage in stages:
            print(f"  {stage}")

            try:
                STAGES[stage](run_id)

            except RuntimeError as e:
                print(f"    [FAILED] {e}")
                failed.append((run_id, f"{stage}: {e}"))
                break

    return finish("03 custom processing", failed)


if __name__ == "__main__":
    sys.exit(main())
