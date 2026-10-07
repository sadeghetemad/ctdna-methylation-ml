"""
Step 05: read-level and fragment features from the BAMs of step 03.

Step 04 averages all CpG calls of a region into one methylation fraction.
That hides the few tumour-derived fragments in early-stage plasma. This
step keeps the fragment as the unit: the CpG calls of both mates of a
read pair are joined into one fragment (where the mates overlap, the
first mate read wins), and only the CpGs inside the region count.

End-repair artifact: the ELSA adapters end in random bases, so the first
~8 bp of every R2 (and the last bases of R1 on short fragments, which
read into the same end) are filled in without bisulfite conversion and
look methylated (~90% in CpG and CH context, against ~6% and ~0.5%
inside the reads). All calls within IGNORE_R2_END bp of the R2 start of
the fragment are therefore ignored, in both mates.

Per region (sample x region matrices; NaN without any fragment):
    meth             methylated / all CpG calls, as step 04 but without the
                     artifact calls; n_calls is its CpG call count
    frac_meth        fragments with >= MIN_CPG CpGs, of which >= 80% methylated
    frac_meth_short  the same, among fragments <= SHORT_MAX bp
    mhl              methylation haplotype load (Guo et al. 2017): weighted
                     fraction of fully methylated runs of 1..k CpGs
    pdr              proportion of discordant reads (Landan et al. 2012):
                     fragments with >= 4 CpGs and both methylated and
                     unmethylated calls
    entropy          Shannon entropy of the 4-CpG patterns (Xie et al. 2011),
                     sliding windows, scaled to [0, 1]
    frac_short       fragments <= SHORT_MAX bp among all fragments
    n_calls, n_frag_all, n_frag, n_frag4, n_frag_short
                     the fragment counts behind these values

Per sample (one table):
    frag_median, frag_frac_short, frag_short_long_ratio, n_fragments
                     fragment lengths of all proper pairs in the BAM
    ch_filtered_fraction
                     reads dropped by the non-conversion filter (QC)
    r2_end_ch_meth   methylated share of the CH calls in the first
                     IGNORE_R2_END bp of R2: strength of the artifact (QC)

Methylation calls come from Bismark's XM tag. BAMs without it (bwa-meth in
nfcore-gpu mode) are called from the read sequence against hg19: on the
C->T converted strand (bwa-meth YD:f, Bismark XG:CT) a C read at a reference
C is methylated and a T unmethylated; on the G->A strand the same for G / A.

Read filters, the same in all modes: proper pair, MAPQ >= MIN_MAPQ, no
secondary / supplementary / duplicate / QC-fail alignments, and at most
MAX_METH_CH methylated non-CpG calls per read (incomplete bisulfite
conversion, as Bismark's filter_non_conversion). nf-core BAMs have no
MAPQ filter of their own, so this also makes the modes comparable.

Each run is cached in features/[<mode>/]read_features/<run>.*; a run is
processed again when its BAM or the parameters change.

Examples:
    python scripts/05_read_features.py
    python scripts/05_read_features.py --strict
"""

import argparse
import json
from multiprocessing import Pool
import re
import sys

import numpy as np
import pandas as pd
import pysam

from utils import (
    COHORT_FILE, COHORT, TARGET_BED, EXPECTED_REGIONS, THREADS,
    BAM_DIR, BAM_SUFFIX, FEATURES_DIR, READ_FEATURES, READ_COUNTS, FASTA,
    IGNORE_R2_END, SAMPLE_FEATURES_FILE, read_feature_file,
)


MIN_MAPQ = 20
MAX_METH_CH = 3     # methylated CHG / CHH calls per read, outside the ignored end
MIN_CPG = 3         # CpGs a fragment needs for frac_meth
METH_CUTOFF = 0.8   # methylated share of a "methylated fragment"
SHORT_MAX = 150     # bp; ctDNA is enriched below the ~167 bp nucleosome peak
ENTROPY_WINDOW = 4
MAX_FRAGMENT = 1000

PARAMS = {
    "min_mapq": MIN_MAPQ, "max_meth_ch": MAX_METH_CH, "min_cpg": MIN_CPG,
    "meth_cutoff": METH_CUTOFF, "short_max": SHORT_MAX,
    "entropy_window": ENTROPY_WINDOW, "ignore_r2_end": IGNORE_R2_END, "version": 3,
}

REGION_COLUMNS = list(READ_FEATURES) + list(READ_COUNTS)
META_COLUMNS = ["run_id", "sample_id", "ml_label"]

CACHE_DIR = FEATURES_DIR / "read_features"
SAMPLE_OUT = SAMPLE_FEATURES_FILE
MISSING_OUT = FEATURES_DIR / f"{COHORT}_read_missing_samples.tsv"

CPG_CALL = re.compile(r"[Zz]")
METH_CH_CALL = re.compile(r"[XH]")


def parse_args():

    p = argparse.ArgumentParser(
        description="Read-level methylation and fragment features from the BAMs."
    )
    p.add_argument("--strict", action="store_true",
                   help="Fail if any cohort run has no BAM file.")
    return p.parse_args()


# ============================================================
# Reads
# ============================================================

class Genome:
    """
    hg19 sequence of one chromosome at a time, for BAMs without an XM tag;
    the BAMs are read in coordinate order, so each chromosome loads once.
    """

    def __init__(self):
        self.fasta = None
        self.chrom = None
        self.seq = None

    def get(self, chrom):
        if chrom != self.chrom:
            if self.fasta is None:
                self.fasta = pysam.FastaFile(str(FASTA))
            self.seq = np.frombuffer(self.fasta.fetch(chrom).upper().encode(), dtype=np.uint8)
            self.chrom = chrom
        return self.seq


GENOME = Genome()     # one per worker process
A, C, G, T, N = (ord(b) for b in "ACGTN")


def methylation_calls(read):
    """Bismark XM string: Z/z methylated / unmethylated CpG, X/x H/h other contexts."""
    if read.has_tag("XM"):
        return read.get_tag("XM")
    return calls_from_sequence(read)


ALIGNED, QUERY_ONLY, REFERENCE_ONLY = (0, 7, 8), (1, 4), (2, 3)   # M = X / I S / D N


def aligned_positions(read):
    """Query and reference positions of the aligned bases, from the CIGAR."""

    q_parts, r_parts = [], []
    qpos, rpos = 0, read.reference_start

    for op, length in read.cigartuples:
        if op in ALIGNED:
            q_parts.append(np.arange(qpos, qpos + length))
            r_parts.append(np.arange(rpos, rpos + length))
            qpos += length
            rpos += length
        elif op in QUERY_ONLY:
            qpos += length
        elif op in REFERENCE_ONLY:
            rpos += length

    if not q_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    return np.concatenate(q_parts), np.concatenate(r_parts)


def calls_from_sequence(read):
    """
    XM-like string for aligners without the XM tag (bwa-meth). CHG and CHH
    are both written as H/h; only CpG and "methylated non-CpG" are used.
    """

    seq = GENOME.get(read.reference_name)
    xm = np.full(read.query_length, ord("."), dtype=np.uint8)

    q, r = aligned_positions(read)
    if len(q) == 0:
        return xm.tobytes().decode()

    base = np.frombuffer(read.query_sequence.encode(), dtype=np.uint8)[q]
    ref = seq[r]

    if read.has_tag("YD"):                      # bwa-meth: f = C->T, r = G->A
        top = read.get_tag("YD") == "f"
    else:                                       # Bismark genome conversion
        top = read.get_tag("XG") == "CT"

    if top:
        neighbour = seq[np.minimum(r + 1, len(seq) - 1)]
        site = (ref == C) & ((base == C) | (base == T)) & (neighbour != N)
        methylated = base == C
    else:
        neighbour = seq[np.maximum(r - 1, 0)]
        site = (ref == G) & ((base == G) | (base == A)) & (neighbour != N)
        methylated = base == G

    cpg = neighbour == (G if top else C)
    code = np.where(cpg, np.where(methylated, ord("Z"), ord("z")),
                    np.where(methylated, ord("H"), ord("h")))
    xm[q[site]] = code[site]

    return xm.tobytes().decode()


def usable(read):
    """Alignment filters; the non-conversion filter is converted()."""
    return (
        read.is_proper_pair
        and not (read.is_secondary or read.is_supplementary
                 or read.is_duplicate or read.is_qcfail)
        and read.mapping_quality >= MIN_MAPQ
    )


def r2_start(read):
    """Genome position of the 5' end of R2, i.e. the filled-in fragment end."""

    left = min(read.reference_start, read.next_reference_start)
    right = left + abs(read.template_length) - 1
    r2_reverse = read.is_reverse if read.is_read2 else read.mate_is_reverse

    return right if r2_reverse else left


def trusted_calls(read, pattern):
    """
    (genome position, call) of the calls matching pattern, without the
    calls within IGNORE_R2_END bp of the R2 start of the fragment.
    """

    xm = methylation_calls(read)
    positions = read.get_reference_positions(full_length=True)
    end = r2_start(read)

    calls = []
    for m in pattern.finditer(xm):
        pos = positions[m.start()]
        if pos is not None and abs(pos - end) >= IGNORE_R2_END:
            calls.append((pos, m.group()))

    return calls


def converted(read):
    return len(trusted_calls(read, METH_CH_CALL)) <= MAX_METH_CH


def load_targets():

    targets = pd.read_csv(TARGET_BED, sep="\t", header=None,
                          names=["chrom", "start0", "end", "region_id"])
    targets["chrom"] = [c if str(c).startswith("chr") else f"chr{c}" for c in targets["chrom"]]

    if len(targets) != EXPECTED_REGIONS:
        raise ValueError(f"Expected {EXPECTED_REGIONS} target regions, found {len(targets)}")

    return targets


def region_fragments(bam, chrom, start0, end, qc):
    """
    CpG calls (0/1, in genome order) and length of each fragment in a region.
    BED coordinates: start0 is 0-based, end exclusive, like pysam positions.
    """

    fragments = {}

    for read in bam.fetch(chrom, start0, end):

        if not usable(read):
            continue

        qc["reads"] += 1

        if not converted(read):
            qc["ch_filtered"] += 1
            continue

        calls, _ = fragments.setdefault(
            read.query_name, ({}, abs(read.template_length))
        )

        for pos, call in trusted_calls(read, CPG_CALL):
            if start0 <= pos < end:
                calls.setdefault(pos, call == "Z")

    return [
        (np.array([calls[p] for p in sorted(calls)], dtype=bool), length)
        for calls, length in fragments.values()
    ]


# ============================================================
# Region features
# ============================================================

def methylated_runs(v):
    """Lengths of the runs of consecutive methylated CpGs."""
    padded = np.concatenate([[False], v, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[1::2] - edges[::2]


def region_features(fragments):

    lengths = np.array([length for _, length in fragments], dtype=np.int64)
    sized = lengths > 0
    short = sized & (lengths <= SHORT_MAX)

    k = np.array([len(v) for v, _ in fragments], dtype=np.int64)
    m = np.array([v.sum() for v, _ in fragments], dtype=np.int64)

    out = dict.fromkeys(REGION_COLUMNS, np.nan)

    out["n_calls"] = int(k.sum())
    if out["n_calls"]:
        out["meth"] = m.sum() / out["n_calls"]

    out["n_frag_all"] = int(sized.sum())
    if sized.any():
        out["frac_short"] = short[sized].mean()

    # frac_meth / frac_meth_short
    informative = k >= MIN_CPG
    methylated = informative & (m >= METH_CUTOFF * k)

    out["n_frag"] = int(informative.sum())
    out["n_frag_short"] = int((informative & short).sum())

    if out["n_frag"]:
        out["frac_meth"] = methylated.sum() / out["n_frag"]
    if out["n_frag_short"]:
        out["frac_meth_short"] = (methylated & short).sum() / out["n_frag_short"]

    # pdr
    four = k >= ENTROPY_WINDOW
    out["n_frag4"] = int(four.sum())

    if out["n_frag4"]:
        out["pdr"] = ((m > 0) & (m < k))[four].mean()

    # mhl: for runs of l = 1..L CpGs, the share of fully methylated runs
    # among all runs of that length, weighted by l
    if k.max(initial=0) > 0:
        max_len = k.max()
        total = np.zeros(max_len + 1)
        meth = np.zeros(max_len + 1)
        lens = np.arange(max_len + 1)

        for v, _ in fragments:
            n = len(v)
            if n == 0:
                continue
            total[1:n + 1] += n - lens[1:n + 1] + 1
            for r in methylated_runs(v):
                meth[1:r + 1] += r - lens[1:r + 1] + 1

        seen = total > 0
        out["mhl"] = (lens[seen] * meth[seen] / total[seen]).sum() / lens[seen].sum()

    # entropy of the sliding 4-CpG patterns
    if out["n_frag4"]:
        weights = 1 << np.arange(ENTROPY_WINDOW)
        patterns = np.zeros(2 ** ENTROPY_WINDOW)

        for v, _ in fragments:
            if len(v) >= ENTROPY_WINDOW:
                windows = np.lib.stride_tricks.sliding_window_view(v, ENTROPY_WINDOW)
                patterns += np.bincount(windows @ weights, minlength=len(patterns))

        p = patterns[patterns > 0] / patterns.sum()
        out["entropy"] = float(-(p * np.log2(p)).sum() / ENTROPY_WINDOW)

    return out


# ============================================================
# Sample features
# ============================================================

def fragment_lengths(bam, qc):
    """
    Lengths of all proper pairs, counted once through read 1; from read 2,
    the CH methylation of its first IGNORE_R2_END bp (artifact QC).
    """

    counts = np.zeros(MAX_FRAGMENT + 1, dtype=np.int64)

    for read in bam.fetch():

        if not usable(read):
            continue

        if read.is_read2:
            xm = methylation_calls(read)
            head = xm[-IGNORE_R2_END:] if read.is_reverse else xm[:IGNORE_R2_END]
            qc["r2_end_ch"] += sum(head.count(c) for c in "xhXH")
            qc["r2_end_ch_meth"] += head.count("X") + head.count("H")

        elif converted(read):
            length = abs(read.template_length)
            if 0 < length <= MAX_FRAGMENT:
                counts[length] += 1

    return counts


def sample_features(counts, qc):

    n = counts.sum()
    cumulative = np.cumsum(counts)

    return {
        "n_fragments": int(n),
        "frag_median": int(np.searchsorted(cumulative, n / 2)) if n else np.nan,
        "frag_frac_short": counts[:SHORT_MAX + 1].sum() / n if n else np.nan,
        # DELFI-style ratio of short (100-150 bp) to long (151-220 bp) fragments
        "frag_short_long_ratio": counts[100:151].sum() / max(counts[151:221].sum(), 1),
        "ch_filtered_fraction": qc["ch_filtered"] / qc["reads"] if qc["reads"] else np.nan,
        "r2_end_ch_meth": qc["r2_end_ch_meth"] / qc["r2_end_ch"] if qc["r2_end_ch"] else np.nan,
    }


# ============================================================
# One run (cached)
# ============================================================

def bam_path(run_id):
    return BAM_DIR / f"{run_id}{BAM_SUFFIX}"


def cache_paths(run_id):
    return CACHE_DIR / f"{run_id}.regions.tsv.gz", CACHE_DIR / f"{run_id}.json"


def cache_key(run_id):
    stat = bam_path(run_id).stat()
    return {"params": PARAMS, "bam": str(bam_path(run_id)),
            "bam_size": stat.st_size, "bam_mtime": int(stat.st_mtime)}


def cached(run_id):

    regions_file, info_file = cache_paths(run_id)

    if not (regions_file.exists() and info_file.exists()):
        return False

    return json.loads(info_file.read_text()).get("key") == cache_key(run_id)


def process_run(run_id):
    """Worker: all regions and the sample features of one run."""

    targets = load_targets()
    qc = {"reads": 0, "ch_filtered": 0, "r2_end_ch": 0, "r2_end_ch_meth": 0}
    rows = []

    with pysam.AlignmentFile(bam_path(run_id)) as bam:

        for chrom, start0, end in zip(targets["chrom"], targets["start0"], targets["end"]):
            if chrom in bam.references:
                rows.append(region_features(region_fragments(bam, chrom, start0, end, qc)))
            else:
                rows.append(dict.fromkeys(REGION_COLUMNS, np.nan))

        sample = sample_features(fragment_lengths(bam, qc), qc)

    regions = pd.DataFrame(rows, columns=REGION_COLUMNS)
    regions.insert(0, "region_id", targets["region_id"].values)

    regions_file, info_file = cache_paths(run_id)
    tmp = regions_file.with_name(regions_file.name.replace(".tsv.gz", ".tmp.tsv.gz"))
    regions.to_csv(tmp, sep="\t", index=False)
    tmp.replace(regions_file)
    info_file.write_text(json.dumps({"key": cache_key(run_id), "sample": sample}, indent=2))

    return run_id, sample


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    cohort = pd.read_csv(COHORT_FILE, sep="\t")
    cohort["run_id"] = cohort["run_id"].astype(str)

    present = cohort[[bam_path(r).exists() for r in cohort["run_id"]]]
    missing = cohort[~cohort["run_id"].isin(present["run_id"])]

    print(f"BAM files: {BAM_DIR} (*{BAM_SUFFIX})")
    print(f"Cohort runs with a BAM: {len(present)}/{len(cohort)}")

    if not missing.empty:
        missing[[c for c in META_COLUMNS if c in cohort.columns]].to_csv(
            MISSING_OUT, sep="\t", index=False)
        print(f"WARNING: no BAM for {', '.join(missing['run_id'])}; "
              f"listed in {MISSING_OUT.name}")
        if args.strict:
            return 1
    else:
        MISSING_OUT.unlink(missing_ok=True)

    if present.empty:
        sys.exit("No BAM files; run step 03 first.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    todo = [r for r in present["run_id"] if not cached(r)]
    print(f"Cached: {len(present) - len(todo)}, to process: {len(todo)}")

    if todo:
        with Pool(min(THREADS, len(todo))) as pool:
            for n, (run_id, sample) in enumerate(pool.imap_unordered(process_run, todo), 1):
                print(f"  [{n}/{len(todo)}] {run_id}: {sample['n_fragments']:,} fragments, "
                      f"median {sample['frag_median']} bp, "
                      f"CH-filtered {sample['ch_filtered_fraction']:.2%}, "
                      f"R2-end CH methylation {sample['r2_end_ch_meth']:.1%}", flush=True)

    # --------------------------------------------------------
    # Matrices in cohort order
    # --------------------------------------------------------

    meta = present[[c for c in META_COLUMNS if c in present.columns]].reset_index(drop=True)
    per_run = [pd.read_csv(cache_paths(r)[0], sep="\t", index_col="region_id")
               for r in present["run_id"]]
    region_ids = per_run[0].index

    for name in REGION_COLUMNS:
        values = pd.DataFrame([df.loc[region_ids, name].to_numpy() for df in per_run],
                              columns=region_ids)
        pd.concat([meta, values], axis=1).to_csv(read_feature_file(name), sep="\t", index=False)

    samples = pd.DataFrame([json.loads(cache_paths(r)[1].read_text())["sample"]
                            for r in present["run_id"]])
    pd.concat([meta, samples], axis=1).to_csv(SAMPLE_OUT, sep="\t", index=False)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("READ FEATURES COMPLETE")
    print("=" * 70)
    print(f"Samples: {len(meta)}, regions: {len(region_ids)}")

    for name, support in READ_FEATURES.items():
        values = pd.read_csv(read_feature_file(name), sep="\t").drop(columns=meta.columns)
        print(f"  {name:<16} observed {values.notna().mean().mean():6.1%}   "
              f"median {np.nanmedian(values.to_numpy()):.4f}   (support: {support})")

    print(f"Saved: {read_feature_file('<feature>').name} and {SAMPLE_OUT.name} in {FEATURES_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
