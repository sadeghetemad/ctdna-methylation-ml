from pathlib import Path
import subprocess
import urllib.request
import gzip
import shutil

ROOT = Path(__file__).resolve().parents[1]

REF_DIR = ROOT / "reference/hg19"
REF_DIR.mkdir(parents=True, exist_ok=True)

GZ = REF_DIR / "hg19.fa.gz"
FA = REF_DIR / "hg19.fa"

URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"


def main():

    if not FA.exists():

        if not GZ.exists():

            print("[DOWNLOAD] hg19")

            urllib.request.urlretrieve(
                URL,
                GZ,
            )

        print("[EXTRACT] hg19.fa")

        with gzip.open(GZ, "rb") as src:
            with open(FA, "wb") as dst:
                shutil.copyfileobj(src, dst)

    else:
        print("[EXISTS] hg19.fa")

    index_marker = REF_DIR / "Bisulfite_Genome"

    if index_marker.exists():
        print("[EXISTS] Bismark genome index")
        return

    print("[BISMARK] Preparing genome")

    subprocess.run(
        [
            "bismark",
            "prepare",
            "--bowtie2",
            "--parallel", "4",
            "--genomic_composition",
            str(REF_DIR),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()