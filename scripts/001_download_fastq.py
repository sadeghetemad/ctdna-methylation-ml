from pathlib import Path
import hashlib
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

COHORT_FILE = ROOT / "metadata/test_cohort_10.tsv"
FASTQ_DIR = ROOT / "data/fastq"
LOG_DIR = ROOT / "logs"

FASTQ_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ENA_API = "https://www.ebi.ac.uk/ena/portal/api/filereport"


def md5sum(path, chunk_size=8 * 1024 * 1024):
    md5 = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            md5.update(chunk)

    return md5.hexdigest()


def get_ena_fastq_info(run_id):
    params = {
        "accession": run_id,
        "result": "read_run",
        "fields": "run_accession,fastq_ftp,fastq_md5,fastq_bytes",
        "format": "tsv",
    }

    url = ENA_API + "?" + urllib.parse.urlencode(params)

    print(f"[ENA] Querying {run_id}")

    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8")

    log_file = LOG_DIR / "ena" / f"{run_id}_ena.tsv"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(text)

    lines = text.strip().splitlines()

    if len(lines) < 2:
        raise RuntimeError(
            f"ENA returned no FASTQ information for {run_id}"
        )

    header = lines[0].split("\t")
    values = lines[1].split("\t")

    row = dict(zip(header, values))

    ftp_urls = row["fastq_ftp"].split(";")
    md5_values = row["fastq_md5"].split(";")
    byte_values = row["fastq_bytes"].split(";")

    if len(ftp_urls) != 2:
        raise RuntimeError(
            f"{run_id}: expected 2 paired FASTQ files, "
            f"but ENA returned {len(ftp_urls)}"
        )

    files = []

    for ftp, expected_md5, size in zip(
        ftp_urls,
        md5_values,
        byte_values,
    ):
        ftp = ftp.strip()

        if not ftp.startswith("http"):
            url = "https://" + ftp
        else:
            url = ftp

        filename = Path(
            urllib.parse.urlparse(url).path
        ).name

        files.append({
            "url": url,
            "filename": filename,
            "md5": expected_md5.strip(),
            "bytes": int(size),
        })

    return files


def download_file(url, destination, expected_bytes):
    if destination.exists():

        actual_size = destination.stat().st_size

        if actual_size == expected_bytes:
            print(
                f"[EXISTS] {destination.name} "
                f"({actual_size / 1024**3:.2f} GB)"
            )
            return

        print(
            f"[INCOMPLETE] {destination.name}: "
            f"{actual_size} / {expected_bytes} bytes"
        )

        destination.unlink()

    print(f"[DOWNLOAD] {destination.name}")
    print(f"           {url}")

    temp_file = destination.with_suffix(
        destination.suffix + ".part"
    )

    if temp_file.exists():
        temp_file.unlink()

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ctDNA-ELSA-pipeline/1.0"
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=120,
    ) as response:

        total = 0
        last_print = time.time()

        with open(temp_file, "wb") as out:

            while True:
                chunk = response.read(
                    8 * 1024 * 1024
                )

                if not chunk:
                    break

                out.write(chunk)
                total += len(chunk)

                if time.time() - last_print >= 5:

                    percent = (
                        total / expected_bytes * 100
                        if expected_bytes
                        else 0
                    )

                    print(
                        f"           "
                        f"{total / 1024**3:.2f} GB "
                        f"({percent:.1f}%)",
                        flush=True,
                    )

                    last_print = time.time()

    actual_size = temp_file.stat().st_size

    if actual_size != expected_bytes:
        raise RuntimeError(
            f"Size mismatch for {destination.name}: "
            f"expected {expected_bytes}, "
            f"downloaded {actual_size}"
        )

    temp_file.rename(destination)

    print(
        f"[OK] {destination.name} downloaded "
        f"({actual_size / 1024**3:.2f} GB)"
    )


def verify_md5(path, expected_md5):
    print(f"[MD5] {path.name}")

    actual_md5 = md5sum(path)

    if actual_md5.lower() != expected_md5.lower():
        # Remove the corrupt file; otherwise the size check in
        # download_file() would accept it again on every rerun.
        path.unlink()

        raise RuntimeError(
            f"MD5 FAILED for {path.name}\n"
            f"Expected: {expected_md5}\n"
            f"Actual:   {actual_md5}"
        )

    print(f"[MD5 OK] {actual_md5}")


def process_run(run_id):
    print()
    print("=" * 70)
    print(f"RUN: {run_id}")
    print("=" * 70)

    files = get_ena_fastq_info(run_id)

    for info in files:

        destination = (
            FASTQ_DIR / info["filename"]
        )

        download_file(
            info["url"],
            destination,
            info["bytes"],
        )

        verify_md5(
            destination,
            info["md5"],
        )


def main():
    if not COHORT_FILE.exists():
        raise FileNotFoundError(
            f"Missing cohort file: {COHORT_FILE}"
        )

    cohort = pd.read_csv(
        COHORT_FILE,
        sep="\t",
    )

    if "run_id" not in cohort.columns:
        raise ValueError(
            "test_cohort_10.tsv has no run_id column"
        )

    runs = (
        cohort["run_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    print(f"Number of runs: {len(runs)}")

    failed_runs = []

    for number, run_id in enumerate(runs, start=1):

        print(
            f"\n######## {number}/{len(runs)} ########"
        )

        try:
            process_run(run_id)

        except Exception as e:
            print()
            print(f"[FAILED] {run_id}")
            print(f"[ERROR] {e}")
            print("[SKIP] Moving to next run...")

            failed_runs.append({
                "run_id": run_id,
                "error": str(e),
            })

            continue

    print()
    print("=" * 70)
    print("DOWNLOAD FINISHED")
    print("=" * 70)

    successful = len(runs) - len(failed_runs)

    print(f"Successful runs: {successful}")
    print(f"Failed runs:     {len(failed_runs)}")

    failed_file = LOG_DIR / "failed_downloads.tsv"

    if failed_runs:

        pd.DataFrame(failed_runs).to_csv(
            failed_file,
            sep="\t",
            index=False,
        )

        print()
        print("Failed runs:")

        for item in failed_runs:
            print(
                f"  {item['run_id']} -> "
                f"{item['error']}"
            )

        print()
        print(
            f"Failure log: {failed_file}"
        )

        return 1

    failed_file.unlink(missing_ok=True)

    print()
    print("ALL RUNS DOWNLOADED AND VERIFIED")

    return 0


if __name__ == "__main__":
    sys.exit(main())
