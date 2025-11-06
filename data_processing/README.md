# ClinVar Data Processing Pipeline

This directory contains scripts and utilities for processing ClinVar variant data for machine learning applications.

## Overview

The pipeline downloads, processes, and prepares ClinVar variant data, including:
- Variant summary data filtering
- VCF file parsing for missense variants
- Gene mapping (NM → NP transcript mappings)
- Protein sequence integration
- Machine learning dataset preparation with train/validation/test splits

## Files

- **`data.py`** - Main data processing pipeline
- **`utils.py`** - Utility functions for Spark setup and data download
- **`const.py`** - Constants and filters for data processing
- **`test_data.py`** - Scripts for testing and data exploration

## Requirements

- Python 3.10+
- PySpark
- pysam (for VCF parsing)
- BioPython (for FASTA parsing)
- pandas
- numpy
- pyarrow

## Installation

```bash
# Create virtual environment
python3.10 -m venv ~/.venvs/clinvar_test

# Activate virtual environment
source ~/.venvs/clinvar_test/bin/activate

# Install dependencies
pip install pyspark pysam biopython pandas numpy pyarrow
```

## Usage

### Basic Usage

Run the pipeline with minimum configuration:

```bash
python data.py --venv-python /path/to/venv/bin/python3.10
```

### Download Data from ClinVar FTP

Automatically download and unpack all necessary files:

```bash
python data.py \
  --venv-python ~/.venvs/clinvar_test/bin/python3.10 \
  --download-data
```

This downloads:
- `variant_summary.txt.gz` - ClinVar variant summary
- `clinvar.vcf.gz` - ClinVar VCF file with variant annotations
- `gene2accession.gz` - Gene mapping file (NM to NP)
- `GCF_000001405.40_GRCh38.p14_protein.faa.gz` - RefSeq protein sequences

### Include Protein Sequences

Add protein sequences and generate mutated sequences:

```bash
python data.py \
  --venv-python ~/.venvs/clinvar_test/bin/python3.10 \
  --join-sequences \
  --sequences-file output_sequences.parquet
```

### Prepare ML Dataset

Create train/validation/test splits with pathogenicity context vectors:

```bash
python data.py \
  --venv-python ~/.venvs/clinvar_test/bin/python3.10 \
  --prepare-ml-dataset
```

### Full Pipeline

Run complete pipeline with all features:

```bash
python data.py \
  --venv-python ~/.venvs/clinvar_test/bin/python3.10 \
  --download-data \
  --join-sequences \
  --prepare-ml-dataset
```

## Command Line Arguments

### Required Arguments

- `--venv-python` - Path to Python executable in virtual environment

### Optional Arguments

- `--download-data` - Download necessary data files from ClinVar FTP
- `--join-sequences` - Join protein sequences and create mutated sequences
- `--sequences-file` - Path to protein sequences parquet file (default: `output_sequences.parquet`)
- `--prepare-ml-dataset` - Prepare machine learning dataset with train/val/test splits

### Help

```bash
python data.py --help
```

## Data Processing Steps

### 1. Variant Summary Processing

Filters ClinVar variant summary data based on:
- Type: Single nucleotide variants (SNVs)
- Assembly: GRCh38
- Review status: Criteria provided, expert panel reviewed, or practice guideline
- Clinical significance: Pathogenic, Likely pathogenic, Benign, Likely benign

Extracts features:
- Variant name and gene symbol
- Transcript IDs (NM and NP)
- Amino acid changes (REF → ALT)
- Position in protein sequence
- Clinical significance (pathogenic = 1, benign = 0)

### 2. VCF File Processing

Parses ClinVar VCF file to extract:
- Variation IDs for missense variants
- Molecular consequence annotations

### 3. Gene Mapping

Maps NM (mRNA) transcript IDs to NP (protein) transcript IDs using NCBI gene2accession file.

### 4. Protein Sequence Integration (Optional)

- Parses RefSeq protein FASTA file
- Joins protein sequences to variant data
- Generates mutated sequences by substituting amino acids

### 5. ML Dataset Preparation (Optional)

Creates training, validation, and test datasets with:
- Sparse mutation vectors (position in sequence)
- Context vectors (pathogenic variants in same protein/phenotype)
- Phenotype filtering (removes "not provided" and "not specified")
- Train/Val/Test split (70%/15%/15% by default)

## Output Files

### Intermediate Files

- `clinvar_missense_ids.parquet` - Missense variant IDs from VCF
- `clinvar_mapping.parquet` - NM to NP transcript mappings
- `output_sequences.parquet` - Protein sequences from FASTA

### Final Datasets

- `full_variation_dataset.parquet` - Complete variant dataset with features
- `df_train_unique_*.parquet` - Training dataset with ML features
- `df_val_unique_*.parquet` - Validation dataset with ML features
- `df_test_unique_*.parquet` - Test dataset with ML features

## Output Schema

### Final Variation Dataset

| Column | Type | Description |
|--------|------|-------------|
| name | string | Variant name from ClinVar |
| gene_symbol | string | Gene symbol |
| phenotype_ids | string | Associated phenotype IDs |
| phenotype_list | string | List of phenotypes (separated by \|) |
| NM_transcript | string | RefSeq mRNA accession (e.g., NM_000546) |
| NP_transcript | string | RefSeq protein accession (e.g., NP_000537) |
| REF_amino_acid | string | Reference amino acid (single letter) |
| ALT_amino_acid | string | Alternate amino acid (single letter) |
| position | string | Position in protein sequence |
| is_pathogenic | int | Clinical significance (1=pathogenic, 0=benign) |
| sequence* | string | Original protein sequence (if --join-sequences) |
| mutated_sequence* | string | Mutated protein sequence (if --join-sequences) |

*Only included when `--join-sequences` flag is used

### ML Dataset

| Column | Type | Description |
|--------|------|-------------|
| name | string | Variant name |
| NP_transcript | string | Protein accession |
| position | string | Position in sequence |
| phenotype_list | string | Associated phenotype |
| is_pathogenic | int | Clinical significance |
| mutation_vector | array<float> | Sparse vector of mutation position |
| context_vector | array<int> | Pathogenic variants in same protein/phenotype |

## Utility Functions

### Download from ClinVar FTP

```python
from utils import download_from_clinvar_ftp

# Download and unpack automatically
file_path = download_from_clinvar_ftp(
    "pub/clinvar/vcf_GRCh38/clinvar.vcf.gz",
    output_dir="./data",
    unpack=True
)
```

### Unpack .gz Files

```python
from utils import unpack_gz_file

unpacked_path = unpack_gz_file("data/clinvar.vcf.gz")
```

## Configuration

Edit `const.py` to modify:

- `AMINO_ACIDS` - Three-letter to single-letter amino acid mappings
- `VARIANTS_FILTERS` - Filtering criteria for variants
- `MAX_SEQUENCE_LENGTH` - Maximum protein sequence length for ML vectors

## Testing and Exploration

Use `test_data.py` for data exploration:

```python
from data_processing.utils import setup_spark_environment

spark = setup_spark_environment("/path/to/venv/bin/python3.10")

# Read and explore data
df = spark.read.parquet("full_variation_dataset.parquet")
df.show()
df.printSchema()

# Check for duplicates
df.groupBy("VariationID").count().filter("count > 1").show()

spark.stop()
```

## Notes

- The pipeline uses PySpark for scalable data processing
- All parquet files are written in overwrite mode
- Duplicate VariationIDs are removed during processing
- Context vectors are computed only for training data and optionally applied to validation/test sets
- Maximum sequence length is set to 2048 amino acids by default

## Troubleshooting

### Spark Memory Issues

Increase driver memory in `utils.py`:

```python
.config("spark.driver.memory", "8g")  # Increase from 4g to 8g
```

### Download Failures

Check network connectivity and try manual download:

```bash
curl -L -O https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz
```

### VCF Parsing Errors

Ensure pysam is properly installed:

```bash
pip install --upgrade pysam
```

## License

This project is part of a master's thesis research on variant pathogenicity prediction.

## Contact

For questions or issues, please contact the repository maintainer.
