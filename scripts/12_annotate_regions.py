"""
12_annotate_regions.py: nearest gene for every panel region (hg19), for interpreting the selected features.

Downloads the UCSC hg19 refGene table once, then reports for each region the gene with the closest
transcription start site, the signed distance (negative = upstream) and whether the region overlaps the
promoter window (TSS -2 kb to +500 bp).

Output (results/modelling/12_annotation/): region_genes.tsv
"""

import gzip
import urllib.request

import numpy as np
import pandas as pd

from ml_common import DATA, log, out_dir

URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz"
COLS = ["bin", "name", "chrom", "strand", "txStart", "txEnd", "cdsStart", "cdsEnd", "exonCount", "exonStarts",
        "exonEnds", "score", "name2", "cdsStartStat", "cdsEndStat", "exonFrames"]


def main():
    out = out_dir("12_annotation")
    gz = out / "refGene.hg19.txt.gz"
    if not gz.exists():
        log(f"download {URL}")
        urllib.request.urlretrieve(URL, gz.with_suffix(".part"))
        gz.with_suffix(".part").rename(gz)
    with gzip.open(gz, "rt") as fh:
        genes = pd.read_csv(fh, sep="\t", names=COLS, usecols=["name", "chrom", "strand", "txStart", "txEnd", "name2"])
    genes = genes[genes["chrom"].str.match(r"^chr[0-9XY]+$")]
    genes["tss"] = np.where(genes["strand"] == "+", genes["txStart"], genes["txEnd"])
    regions = pd.read_csv(DATA / "regions.tsv", sep="\t").set_index("region_id")
    rows = []
    for rid, r in regions.iterrows():
        chrom = str(r["chrom"])
        chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
        g = genes[genes["chrom"] == chrom]
        mid = (r["start0"] + r["end"]) / 2
        if g.empty:
            rows.append({"region_id": rid, "gene": "", "tss_distance": np.nan, "promoter": False})
            continue
        d = mid - g["tss"].to_numpy()
        d = np.where(g["strand"].to_numpy() == "+", d, -d)
        k = int(np.argmin(np.abs(d)))
        rows.append({"region_id": rid, "chrom": chrom, "start0": int(r["start0"]), "end": int(r["end"]),
                     "gene": g["name2"].iloc[k], "tss_distance": int(d[k]), "promoter": bool(-2000 <= d[k] <= 500)})
    ann = pd.DataFrame(rows).set_index("region_id")
    ann.to_csv(out / "region_genes.tsv", sep="\t")
    log(f"{len(ann)} regions; {ann['promoter'].mean():.0%} in a promoter window; "
        f"{ann['gene'].nunique()} distinct nearest genes -> {out}")


if __name__ == "__main__":
    main()
