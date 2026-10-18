"""
Put the inputs of the AWS Batch run into S3 (before step 03 --aws-batch).

    fastq       Streams each cohort run's two FASTQ files from ENA straight
                into s3://<bucket>/fastq/, checking size and MD5. Runs
                already in S3 are skipped, so it can be rerun. A FASTQ file
                that is already in data/fastq/ is uploaded from there.
    reference   Uploads hg19.fa and the Bismark index (Bisulfite_Genome) to
                s3://<bucket>/reference/hg19/ (about 17 GB, one time).

The bucket comes from CTDNA_S3_BUCKET (terraform output env_exports).

Examples:
    python scripts/03_aws_stage.py reference
    python scripts/03_aws_stage.py fastq --workers 12
"""

import argparse
import hashlib
import importlib
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import REF_DIR, cohort_runs


CHUNK = 8 * 1024 * 1024


def bucket():
    name = os.environ.get("CTDNA_S3_BUCKET")
    if not name:
        sys.exit("CTDNA_S3_BUCKET is not set. Run:\n"
                 "  source <(terraform -chdir=infra/aws output -raw env_exports)")
    return name


def s3_size(uri):
    """Size in bytes of an S3 object, or None when it does not exist."""
    bucket_name, key = uri[len("s3://"):].split("/", 1)
    result = subprocess.run(
        ["aws", "s3api", "head-object", "--bucket", bucket_name, "--key", key,
         "--query", "ContentLength", "--output", "text"],
        capture_output=True, text=True,
    )
    return int(result.stdout) if result.returncode == 0 else None


def s3_delete(uri):
    subprocess.run(["aws", "s3", "rm", uri, "--only-show-errors"])


def stage_file(f, uri):
    """Stream one ENA FASTQ file into S3 and verify its MD5."""

    if s3_size(uri) == f["bytes"]:
        return "exists"

    md5 = hashlib.md5()
    request = urllib.request.Request(f["url"], headers={"User-Agent": "ctDNA-ELSA-pipeline/1.0"})

    upload = subprocess.Popen(
        ["aws", "s3", "cp", "-", uri, "--expected-size", str(f["bytes"]), "--only-show-errors"],
        stdin=subprocess.PIPE,
    )

    try:
        if f["path"].exists() and f["path"].stat().st_size == f["bytes"]:
            source = open(f["path"], "rb")
        else:
            source = urllib.request.urlopen(request, timeout=120)

        with source:
            while chunk := source.read(CHUNK):
                md5.update(chunk)
                upload.stdin.write(chunk)
    except BaseException:
        upload.kill()
        raise
    finally:
        try:
            upload.stdin.close()
        except BrokenPipeError:
            pass

    if upload.wait() != 0:
        raise RuntimeError(f"aws s3 cp failed for {uri}")

    if md5.hexdigest().lower() != f["md5"].lower():
        s3_delete(uri)
        raise RuntimeError(f"MD5 failed for {f['path'].name}")

    return "uploaded"


def stage_run(run_id, ena_fastq_files, prefix):
    return [(f["path"].name, stage_file(f, f"{prefix}/{f['path'].name}"))
            for f in ena_fastq_files(run_id)]


def stage_fastq(args):

    prefix = f"s3://{bucket()}/fastq"
    ena_fastq_files = importlib.import_module("01_data").ena_fastq_files
    run_ids = cohort_runs()
    failed = []

    print(f"{len(run_ids)} runs -> {prefix}/")

    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(stage_run, r, ena_fastq_files, prefix): r for r in run_ids}

        for n, future in enumerate(as_completed(futures), start=1):
            run_id = futures[future]
            try:
                states = ", ".join(f"{name}: {state}" for name, state in future.result())
                print(f"[{n}/{len(run_ids)}] {run_id}  {states}", flush=True)
            except Exception as e:
                print(f"[{n}/{len(run_ids)}] {run_id}  FAILED: {e}", flush=True)
                failed.append(run_id)

    if failed:
        print(f"\n{len(failed)} run(s) failed; rerun this command to retry them:")
        print("  " + " ".join(failed))
        return 1

    print(f"\nAll {len(run_ids)} runs are in {prefix}/")
    return 0


def stage_reference(args):

    target = f"s3://{bucket()}/reference/hg19"

    for path in (REF_DIR / "hg19.fa", REF_DIR / "Bisulfite_Genome"):
        if not path.exists():
            sys.exit(f"Missing {path}\nRun: python scripts/02_reference.py")

    cmd = ["aws", "s3", "sync", str(REF_DIR), target, "--only-show-errors",
           "--exclude", "*", "--include", "hg19.fa", "--include", "hg19.fa.fai",
           "--include", "Bisulfite_Genome/*"]

    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main():

    p = argparse.ArgumentParser(description="Stage inputs of the AWS Batch run in S3.")
    sub = p.add_subparsers(dest="what", required=True)

    fastq = sub.add_parser("fastq", help="ENA -> s3://<bucket>/fastq/")
    fastq.add_argument("--workers", type=int, default=8, help="Parallel runs (default: %(default)s).")
    fastq.set_defaults(func=stage_fastq)

    reference = sub.add_parser("reference", help="hg19 and the Bismark index -> s3://<bucket>/reference/hg19/")
    reference.set_defaults(func=stage_reference)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
