"""
Step 02: UCSC hg19 FASTA and its Bismark index (one time, both modes).

nfcore mode reuses the index through --bismark_index, unless
03_process_nfcore.py --build-index is given.
"""

import gzip
import shutil
import sys
import urllib.request

from utils import LOG_DIR, REF_DIR, FASTA, run

URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
FASTA_GZ = REF_DIR / "hg19.fa.gz"
INDEX_DIR = REF_DIR / "Bisulfite_Genome"


def main():

    REF_DIR.mkdir(parents=True, exist_ok=True)

    if FASTA.exists():
        print(f"[EXISTS] {FASTA.name}")

    else:
        if not FASTA_GZ.exists():
            print(f"[DOWNLOAD] {URL}")
            urllib.request.urlretrieve(URL, FASTA_GZ)

        print(f"[EXTRACT] {FASTA.name}")
        tmp = FASTA.with_name(FASTA.name + ".tmp")

        with gzip.open(FASTA_GZ, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)

        tmp.rename(FASTA)

    if INDEX_DIR.exists():
        print("[EXISTS] Bismark genome index")
        return 0

    print("[BISMARK] Preparing genome index")

    try:
        run(
            ["bismark", "prepare", "--bowtie2", "--parallel", "4",
             "--genomic_composition", REF_DIR],
            LOG_DIR / "reference.bismark_prepare.log",
        )

    except RuntimeError as e:
        # A partial index would count as done on the next run
        shutil.rmtree(INDEX_DIR, ignore_errors=True)
        print(f"[FAILED] {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
