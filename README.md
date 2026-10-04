# 🧬 Ultrasensitive ctDNA Detection via Deep Methylation Sequencing and Machine Learning

> An end-to-end pipeline from raw plasma cfDNA bisulfite sequencing reads to machine-learning models for early-stage lung cancer detection, built on the **ELSA** targeted methylation panel.

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![Bismark](https://img.shields.io/badge/Bismark-v3-8A2BE2)
![Genome](https://img.shields.io/badge/genome-hg19-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-≥1.4-F7931E?logo=scikitlearn&logoColor=white)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Pipeline at a Glance](#-pipeline-at-a-glance)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Modes: Classic and nf-core](#-modes-classic-and-nf-core)
- [Scripts](#-scripts)
- [Machine Learning](#-machine-learning)
- [Configuration](#%EF%B8%8F-configuration)
- [Resuming, Logging and Errors](#-resuming-logging-and-errors)
- [Project Structure](#-project-structure)
- [Disk Usage](#-disk-usage)

---

## 🔭 Overview

This repository rebuilds the region-level methylation features used by the **ELSA** targeted bisulfite sequencing panel from public SRA data, then packages them for machine learning.

| | |
|---|---|
| 🎯 **Task** | Stage I lung cancer (`ml_label = 1`) vs. non-cancer control (`ml_label = 0`) |
| 🧪 **Assay** | Targeted bisulfite sequencing of plasma cfDNA, paired-end |
| 📍 **Features** | Coverage-weighted CpG methylation over the published ELSA plasma panel regions |
| 🗺️ **Reference** | UCSC hg19 |

Every step is a small standalone Python script that wraps well-established tools (**FastQC**, **Trim Galore**, **Bismark**, **Bowtie2**, **samtools**). The Python code handles orchestration, feature engineering and QC only.

The read processing (steps 01–07) can run in two **modes**: the `classic` Python scripts, or [nf-core/methylseq](https://nf-co.re/methylseq) (`nfcore`). Both feed the same feature and ML steps.

---

## 🗺️ Pipeline at a Glance

```text
 📋 Metadata            📥 Reads               🔬 Alignment              📊 Features            🤖 ML
┌──────────────┐    ┌──────────────┐    ┌───────────────────┐    ┌───────────────┐    ┌──────────────┐
│ 0  metadata  │    │ 001 download │    │ 04 reference      │    │ 08 features   │    │ 10 ML dataset│
│ 00 cohort    │ ─▶ │ 01  FastQC   │ ─▶ │ 05 align          │ ─▶ │ 09 feature QC │ ─▶ │ notebooks    │
│              │    │ 02  trim     │    │ 06 dedup + filter │    │               │    │              │
│              │    │ 03  FastQC   │    │ 07 methylation    │    │               │    │              │
└──────────────┘    └──────────────┘    └───────────────────┘    └───────────────┘    └──────────────┘
```

---

## 📦 Installation

```bash
git clone https://github.com/sadeghetemad/ctdna-methylation-ml.git
cd ctdna-methylation-ml

conda env create -f environment.yml
conda activate ctdna10
```

> [!IMPORTANT]
> The scripts use the **Bismark v3** sub-commands (`bismark prepare / align / dedup / extract`). Older Bismark 0.x releases are not supported.

**Required input files** (not produced by the pipeline):

| File | Source |
|---|---|
| `metadata/raw/SraRunTable.csv` | SRA Run Selector export for the ELSA BioProject |
| `metadata/publication/ELSA_supplementary_tables.xlsx` | Supplementary tables of the ELSA publication |

---

## 🚀 Quick Start

Run all commands from the project root. To run every step in order:

```bash
python main.py                    # classic mode, all steps, stops at the first failure
python main.py --mode nfcore      # nf-core/methylseq instead of steps 01-07
python main.py --list             # show the steps of a mode
python main.py --from 05          # resume from a step
python main.py --from 08 --to 10  # run a range of steps
```

Or run the steps one by one (classic mode):

```bash
# 📋 Metadata and cohort
python scripts/0_prepare_metadata.py
python scripts/00_select_samples.py

# 📥 Reads and QC
python scripts/001_download_fastq.py
python scripts/01_fastqc.py
python scripts/02_trim.py
python scripts/03_fastqc_trimmed.py

# 🔬 Reference (one time), alignment and methylation calling
python scripts/04_prepare_references.py
python scripts/05_align.py
python scripts/06_dedup_filter.py
python scripts/07_extract_methylation.py

# 📊 Features and QC
python scripts/08_build_features.py
python scripts/09_feature_qc.py

# 🤖 ML-ready dataset
python scripts/10_prepare_ml_dataset.py
```

To set the number of threads (8 by default):

```bash
CTDNA_THREADS=16 python scripts/05_align.py
```

---

## 🔀 Modes: Classic and nf-core

```text
                         classic:  01 FastQC → 02 trim → 03 FastQC → 05 align → 06 dedup + filter → 07 methylation
 0 → 00 → 001 → 04 ─▶                                                                                              ─▶ 08 → 09 → 10
                         nfcore:   nf  nf-core/methylseq (FastQC, Trim Galore, Bismark, dedup, methylation, MultiQC)
```

| | `classic` (default) | `nfcore` |
|---|---|---|
| Steps 01–07 | Python scripts `01`–`07` | `scripts/nf_methylseq.py` (step `nf`) runs nf-core/methylseq **4.2.0** |
| Needs | Tools from `environment.yml` | Nextflow (in `environment.yml`) and Docker, Singularity or Apptainer |
| Coverage files for 08 | `data/methylation/<run>.filtered.namesort.bismark.cov.gz` | `data/nfcore/methylation/<run>.bismark.cov.gz` (links into the nf-core results) |
| Features, QC, ML data | `features/`, `results/feature_qc/`, `notebooks/data/` | the same folders with an `nfcore/` subfolder |

The two modes never overwrite each other, so you can run both and compare `features/` with `features/nfcore/`.

`main.py --mode` sets the `CTDNA_MODE` environment variable for every step. To run a single step in nf-core mode by hand:

```bash
CTDNA_MODE=nfcore python scripts/08_build_features.py
```

### Running in nf-core mode

```bash
# One time: Docker (or Singularity / Apptainer) and Nextflow
sudo apt-get install -y docker.io && sudo usermod -aG docker $USER   # then log in again
conda env update -f environment.yml                                 # adds nextflow

# Whole pipeline
python main.py --mode nfcore                          # Docker
python main.py --mode nfcore --nf-profile singularity # Singularity

# Only nf-core, then the features
python main.py --mode nfcore --from nf --to nf
python main.py --mode nfcore --from 08
```

Run long jobs inside `tmux` or `screen`. If the run stops, run the same command again: Nextflow resumes with `-resume`.

The `nf` step:
1. 📋 writes `data/nfcore/inputs/samplesheet.csv` from the cohort and `data/fastq/<run>_{1,2}.fastq.gz`. If a run has no FASTQ files, it stops and names the run.
2. 🗺️ writes the panel BED with UCSC names (`chr1, …`) to match `hg19.fa`.
3. ▶️ runs nf-core/methylseq with `config/methylseq_params.yaml` and `config/methylseq.config`. If `reference/hg19/Bisulfite_Genome` exists, it is reused through `--bismark_index`.
4. 🔗 links each run's `.bismark.cov.gz` to `data/nfcore/methylation/`.

| `main.py` option | Description |
|---|---|
| `--nf-profile` | Nextflow profile: `docker` (default), `singularity`, `apptainer`, `conda` |
| `--build-index` | Let nf-core build its own Bismark index. Use this if alignment fails with the index from step 04 (built with Bismark v3) |

To check the inputs and print the Nextflow command without running it:

```bash
python scripts/nf_methylseq.py --dry-run
```

### nf-core settings

`config/methylseq_params.yaml` follows the classic scripts as closely as nf-core allows:

| Step | Classic | nf-core |
|---|---|---|
| Trimming | Trim Galore `--illumina -q 20 --length 20` | Trim Galore, Q20, `length_trim: 20`, adapter auto-detected |
| Alignment | Bismark + Bowtie2 | `aligner: bismark` |
| Deduplication | `bismark dedup` | `deduplicate_bismark` |
| Methylation | `bismark extract --comprehensive --cytosine_report` | `comprehensive`, `cytosine_report`, `no_overlap` |
| Ignored bases | none | `ignore_r1/r2`, `ignore_3prime_r1/r2` set to `0` (the nf-core default is 2 bp on R2) |
| Read filter | `samtools view -q 20 -f 2 -F 0x904` | **No MAPQ filter.** Bismark already keeps only unique, concordant pairs, so only the MAPQ ≥ 20 cut is missing |
| QC report | separate FastQC folders | one MultiQC report: `results/nfcore_methylseq/multiqc/multiqc_report.html` |
| Targeted analysis | none | calls restricted to the ELSA regions (`run_targeted_sequencing`) |

Because of the missing MAPQ filter, the features of the two modes are close but not identical.

`config/methylseq.config` limits every nf-core job to **8 CPUs, 28 GB and 72 h**. Edit it to match your machine.

---

## 🧩 Scripts

### 📋 Metadata

#### `0_prepare_metadata.py`: build labels, manifests and target regions

Merges the SRA run table with the publication's supplementary tables. The outputs are:
- clinical labels
- a labeled sample manifest
- the primary Stage I vs. control cohort
- the panel target regions as a 0-based BED file

```bash
python scripts/0_prepare_metadata.py
```

| Output | Description |
|---|---|
| `metadata/ELSA_clinical_labels.tsv` | Per-sample clinical annotations and `ml_label` |
| `metadata/sample_manifest.tsv`, `sample_manifest_labeled.tsv` | SRA runs, without and with labels |
| `metadata/primary_cohort_stage1_vs_control.tsv` | Analysis cohort |
| `metadata/primary_cohort_SRR_Acc_List.txt` | Run accessions of the cohort |
| `metadata/targets/ELSA_plasma_2473_hg19.bed` | Target regions (BED, 0-based) |
| `metadata/targets/ELSA_plasma_2473_regions.tsv` | Target regions in publication coordinates |

#### `00_select_samples.py`: select a balanced working cohort

Draws a class-balanced, reproducible subset (`random_state = 42`) from the primary cohort. Use it to develop and test on a small set before running at scale.

```bash
python scripts/00_select_samples.py
```

**Output:** `metadata/<cohort>.tsv` and `metadata/<cohort>_SRR.txt`

---

### 📥 Reads and Quality Control

#### `001_download_fastq.py`: download paired FASTQ files from ENA

Asks the ENA API for the FASTQ URLs of each run, downloads both mates, and checks the **file size and MD5** of each. A file that fails the check is deleted and reported. Runs that fail are listed in `logs/failed_downloads.tsv`.

```bash
python scripts/001_download_fastq.py
```

**Output:** `data/fastq/<run>_1.fastq.gz`, `data/fastq/<run>_2.fastq.gz`

#### `01_fastqc.py`: QC of the raw reads

```bash
python scripts/01_fastqc.py
```

**Output:** `results/fastqc_raw/*_fastqc.{html,zip}`

#### `02_trim.py`: adapter and quality trimming

Runs Trim Galore in paired mode with Illumina adapters, a Q20 quality cutoff and a 20 bp minimum read length.

```bash
python scripts/02_trim.py
```

**Output:** `data/trimmed/<run>_{1,2}_val_{1,2}.fq.gz` and the trimming reports

#### `03_fastqc_trimmed.py`: QC of the trimmed reads

```bash
python scripts/03_fastqc_trimmed.py
```

**Output:** `results/fastqc_trimmed/`

---

### 🔬 Alignment and Methylation Calling

#### `04_prepare_references.py`: hg19 reference and Bismark index (one time)

Downloads the UCSC hg19 FASTA and builds the bisulfite-converted Bowtie2 index with `bismark prepare`. If the reference or the index already exists, that part is skipped.

```bash
python scripts/04_prepare_references.py
```

**Output:** `reference/hg19/hg19.fa`, `reference/hg19/Bisulfite_Genome/`

#### `05_align.py`: bisulfite alignment

Aligns the paired-end reads with Bismark (Bowtie2 backend).

```bash
python scripts/05_align.py
```

**Output:** `data/aligned/<run>_pe.bam`, `data/aligned/<run>_PE_report.txt`

#### `06_dedup_filter.py`: deduplication and read filtering

1. 🧹 Removes PCR duplicates with `bismark dedup`.
2. 🎚️ Keeps reads with MAPQ ≥ 20 that are in proper pairs (`-f 2`), and drops unmapped, secondary and supplementary alignments (`-F 0x904`).
3. 📑 Writes a coordinate-sorted, indexed BAM and a name-sorted BAM for methylation extraction.

```bash
python scripts/06_dedup_filter.py
```

**Output:** `data/dedup/`, `data/filtered/<run>.filtered.bam(.bai)`, `data/filtered/<run>.filtered.namesort.bam`

#### `07_extract_methylation.py`: per-CpG methylation calls

Runs `bismark extract` on the name-sorted BAMs.

```bash
python scripts/07_extract_methylation.py
```

**Output:** `data/methylation/<run>.filtered.namesort.bismark.cov.gz` (used downstream), plus the CpG report, M-bias and context summaries

---

### 📊 Features

#### `08_build_features.py`: region-level methylation matrix

Sums the CpG calls within each panel region into a **coverage-weighted methylation fraction**:

```text
methylation_fraction = Σ methylated / Σ (methylated + unmethylated)
```

- 📐 A CpG belongs to a region when `BED start + 1 ≤ CpG position ≤ BED end`.
- 🏷️ Chromosome names are normalized to UCSC style (`1` → `chr1`).
- ⚡ Aggregation is vectorized with binary search and cumulative sums.
- 🕳️ A region with no calls is `NaN`.
- ⚠️ Cohort samples without a coverage file are listed in `<cohort>_missing_samples.tsv`. With `--strict`, a missing sample stops the script with an error.

```bash
python scripts/08_build_features.py            # warn about missing samples
python scripts/08_build_features.py --strict   # fail if any sample is missing
```

| Output | Content |
|---|---|
| `features/<cohort>_features.tsv` | Methylation fraction: `run_id, sample_id, ml_label, ELSA_0001 …` |
| `features/<cohort>_coverage.tsv` | Methylated + unmethylated calls per region |
| `features/<cohort>_n_cpg.tsv` | Observed CpG sites per region |

#### `09_feature_qc.py`: feature-matrix QC

Writes QC tables for:
- missingness by sample, region and label
- feature variance
- per-sample methylation and depth summaries

```bash
python scripts/09_feature_qc.py
```

**Output:** `results/feature_qc/*.tsv`

---

### 🤖 ML Dataset

#### `10_prepare_ml_dataset.py`: build the ML-ready dataset

Applies **label-free** filters only:
1. It masks values below a minimum read depth.
2. It drops regions with too many missing values.
3. It drops constant regions.

Imputation, scaling and feature selection are deliberately left to the modelling code, where they run **inside** the cross-validation folds.

```bash
python scripts/10_prepare_ml_dataset.py
python scripts/10_prepare_ml_dataset.py --min-depth 30 --max-missing 0.1 --out-dir notebooks/data
```

| Option | Default | Description |
|---|---|---|
| `--min-depth` | `20` | A value with fewer calls is set to `NaN` |
| `--max-missing` | `0.2` | A region missing in a larger fraction of samples is dropped |
| `--out-dir` | `notebooks/data` | Output directory |

| Output | Content |
|---|---|
| `X_methylation.tsv` | Sample × region methylation matrix |
| `X_coverage.tsv` | Matching read depth |
| `samples.tsv` | Label, clinical covariates, `model_group`, `submission_series`, QC metrics |
| `regions.tsv` | All regions with QC statistics and `kept` / `drop_reason` |
| `dataset_info.json` | Parameters, counts and build timestamp |

> [!NOTE]
> The `predicted` column from the publication (the output of the published classifier) is excluded on purpose, because it would leak the label.

---

## 🤖 Machine Learning

```bash
jupyter lab
```

| Notebook | Description |
|---|---|
| 📓 `notebooks/01_EDA.ipynb` | Missingness, methylation distributions, PCA |
| 📓 `notebooks/02_ML_baseline.ipynb` | Leakage-safe baseline: confounder checks, repeated stratified CV (Dummy, L2 logistic regression, in-fold top-k selection, random forest), permutation test, out-of-fold ROC, per-region statistics |

The model outputs are written to `results/ml/`.

The notebooks read `notebooks/data/`. For the nf-core mode dataset, set `DATA_DIR = ROOT / "notebooks/data/nfcore"` in the notebook.

**Best practices built into the notebooks**

- ✅ Preprocessing runs inside an sklearn `Pipeline`, so it is fit on the training folds only.
- ✅ A permutation test checks performance against chance.
- ✅ Batch and demographic confounders are checked (`submission_series`, age, gender, DNA input).
- ✅ `model_group` gives a held-out train/validation split, as in the original study.

---

## ⚙️ Configuration

| Setting | Where | Default |
|---|---|---|
| Mode | `main.py --mode`, or the `CTDNA_MODE` environment variable (`scripts/utils.py`) | `classic` |
| Threads | `CTDNA_THREADS` environment variable (`scripts/utils.py`) | `8` |
| Cohort file | `COHORT_FILE` in `001`, `08` and `nf_methylseq`, `FEATURE_FILE` / `COVERAGE_FILE` in `09`, `COHORT` in `10` | working cohort from `00` |
| Cohort size / seed | `00_select_samples.py` | balanced, `random_state = 42` |
| MAPQ threshold | `MIN_MAPQ` in `06_dedup_filter.py` | `20` |
| Trimming | `02_trim.py` | Q20, Illumina adapter, min length 20 |
| nf-core parameters | `config/methylseq_params.yaml` | see [nf-core settings](#nf-core-settings) |
| nf-core resources | `config/methylseq.config` | 8 CPUs, 28 GB, 72 h per job |

To run the **full cohort**, point the cohort settings at `metadata/primary_cohort_stage1_vs_control.tsv`.

---

## 🔁 Resuming, Logging and Errors

- ♻️ **Resumable.** A step skips a run only when that run's *final* outputs exist. For example, a filtered BAM counts only if its `.bai` index also exists. Runs that were interrupted are processed again.
- 🛡️ **Atomic writes.** samtools writes to `*.tmp.bam` and renames the file only when it succeeds.
- 📝 **Logs.** Each tool writes to `logs/<run>.<step>.log`, and ENA responses are saved in `logs/ena/`.
- 🚦 **Exit codes.** When a run fails, the step continues with the remaining runs, prints a summary and exits with code `1`. You can chain the steps safely with `&&` or a workflow manager.
- 🔄 **Rerunning.** To rerun a step for a sample, delete that sample's outputs for that step.

---

## 📁 Project Structure

```text
ctdna-methylation-ml/
├── 📄 README.md
├── 📄 environment.yml            # conda environment
├── 📄 main.py                    # runs all steps in order (--mode classic | nfcore)
├── 📂 scripts/                   # numbered pipeline steps
│   ├── nf_methylseq.py           # nfcore mode: nf-core/methylseq for steps 01-07
│   └── utils.py                  # shared paths, mode, threads, logged command runner
├── 📂 config/                    # nf-core/methylseq parameters and resources
├── 📂 metadata/
│   ├── raw/                      # SRA run table (input)
│   ├── publication/              # supplementary tables (input)
│   └── targets/                  # panel regions (BED + TSV)
├── 📂 reference/hg19/            # FASTA + Bismark index
├── 📂 data/                      # large intermediates (not version-controlled)
│   ├── fastq/  trimmed/  aligned/
│   ├── dedup/  filtered/  methylation/
│   └── nfcore/                   # nfcore mode: inputs/, work/, methylation/
├── 📂 features/                  # sample × region matrices (nfcore/ for nfcore mode)
├── 📂 results/
│   ├── fastqc_raw/  fastqc_trimmed/
│   ├── nfcore_methylseq/         # nf-core results and MultiQC report
│   ├── feature_qc/               # (nfcore/ for nfcore mode)
│   └── ml/
├── 📂 notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_ML_baseline.ipynb
│   └── data/                     # ML-ready dataset (nfcore/ for nfcore mode)
└── 📂 logs/
```

---

## 💾 Disk Usage

| Item | Approx. size |
|---|---|
| hg19 FASTA + Bismark index | ~18 GB (one time) |
| Intermediate files per sample | ~10 GB |
| nf-core work directory (`data/nfcore/work/`) | about the same as the classic intermediates; delete it once `results/nfcore_methylseq/` is complete (resuming is then no longer possible) |

Once a sample has its `.bismark.cov.gz`, you can remove these files:

| Path | Needed later? |
|---|---|
| `data/methylation/CHG_*`, `CHH_*` | ❌ Only CpG context is used |
| `data/methylation/*.bedGraph.gz` | ❌ Step 08 reads `.bismark.cov.gz` |
| `reference/hg19/hg19.fa.gz` | ❌ Once `hg19.fa` is extracted |
| `data/filtered/*.namesort.bam` | ⚠️ Only to rerun step 07 |
| `data/trimmed/`, `data/aligned/`, `data/dedup/` | ⚠️ Only to rerun steps 05–07 |

> [!TIP]
> Keep `data/methylation/*.CpG_report.txt.gz`: step 07 checks for it to decide whether a sample is already done.
> Always keep `data/fastq/`, `data/methylation/*.bismark.cov.gz`, `metadata/` and `features/`. Everything else can be regenerated.
