import numpy as np
import pandas as pd
import pysam
import pyspark.sql.functions as F
from Bio import SeqIO
from const import AMINO_ACIDS, MAX_SEQUENCE_LENGTH, VARIANTS_FILTERS
from pyspark.ml.linalg import DenseVector, SparseVector, Vectors, VectorUDT
from pyspark.sql.types import ArrayType, FloatType, IntegerType
from utils import download_from_clinvar_ftp, parse_arguments, setup_spark_environment


def extract_variant_features(df):
    """
    Extract relevant features from ClinVar variant summary DataFrame
    Args:
        df: Input DataFrame with ClinVar variant summary data
    Returns:
        DataFrame with extracted features
    """
    cols = [
        "Name",
        "VariationID",
        "GeneSymbol",
        "ClinSigSimple",
        "PhenotypeIDS",
        "PhenotypeList",
        "ReviewStatus",
    ]

    patterns = {
        "transcription": (F.col("Name"), r"^([^()]+)\(", 1),
        "gene": (F.col("Name"), r"\(([^)]+)\)", 1),
        "SNV": (F.col("Name"), r"p\.([A-Za-z0-9]+)", 1),
        "REF": (F.col("SNV"), r"^([A-Za-z]+)", 1),
        "POS": (F.col("SNV"), r"([0-9]+)", 1),
        "ALT": (F.col("SNV"), r"([A-Za-z]+)$", 1),
    }

    df = df.select(*cols)
    for new_col, (src, regex, group) in patterns.items():
        df = df.withColumn(new_col, F.regexp_extract(src, regex, group))
    df = df.withColumn("ClinSigSimple", F.col("ClinSigSimple").cast(IntegerType()))
    return df.dropDuplicates(["VariationID"])


def process_variant_summary_file(input_file: str):
    """
    Process ClinVar variant summary file to filter SNVs and extract features.
    Args:
        input_file: Path to the ClinVar variant summary file (TSV format)
    Data Schema:

    """
    df = spark.read.option("header", True).option("sep", "\t").csv(input_file)
    condition = (
        (F.col("Type") == VARIANTS_FILTERS["Type"])
        & (F.col("Assembly") == VARIANTS_FILTERS["Assembly"])
        & (F.col("ReviewStatus").isin(VARIANTS_FILTERS["ReviewStatus"]))
        & (F.col("ClinicalSignificance").isin(VARIANTS_FILTERS["ClinicalSignificance"]))
    )
    df_filtered = df.filter(condition)
    df_processed = extract_variant_features(df_filtered)
    return df_processed


def extract_missense_variants_from_vcf(vcf_file: str) -> list:
    """
    Extract VariationIDs of missense variants from a VCF file.
    Args:
        vcf_file: Path to the VCF file
    Returns:
        List of VariationIDs for missense variants
    File Schema:
        https://ftp.ncbi.nlm.nih.gov/pub/clinvar/README_VCF.txt
    """
    missense_ids = []
    vcf_in = pysam.VariantFile(vcf_file)
    for rec in vcf_in:
        mc = rec.info.get("MC")  # Get the MC (Molecular Consequence) annotation
        if (
            mc
            and any("missense_variant" in m for m in mc)
        ):
            missense_ids.append(rec.id)
    vcf_in.close()

    return missense_ids


def process_vcf_file(vcf_file: str, output_path: str = "clinvar_missense_ids.parquet"):
    """
    Process VCF file to extract missense variant IDs and save to Parquet.
    Args:
        vcf_file: Path to the VCF file
        output_path: Path to save the Parquet file with missense variant IDs
    """
    missense_ids = extract_missense_variants_from_vcf(vcf_file)
    ids_num = len(missense_ids)
    print(f"Number of missense variants: {ids_num}")
    print(len(list(set(missense_ids))))
    if ids_num > 0:
        df = spark.createDataFrame([(id,) for id in missense_ids], ["VariationID"])
        df.write.mode("overwrite").parquet(output_path)
        print("Missense variants saved successfully.")
        return df


def process_gene_mappings(input_file: str, output_file: str):
    """Process gene mappings (NM -> NP) from input file and save to output file."""
    df = spark.read.option("header", True).option("sep", "\t").csv(input_file)
    df_filtered = (
        df.filter(F.col("status").isin("REVIEWED", "VALIDATED"))
        .select(
            F.col("`RNA_nucleotide_accession.version`").alias("NM_transcript"),
            F.col("`protein_accession.version`").alias("NP_transcript"),
        )
        .filter((F.col("NM_transcript") != "-") & (F.col("NP_transcript") != "-"))
        .distinct()
    )
    df_filtered.write.mode("overwrite").parquet(output_file)
    print("Mapping saved successfully.")
    return df_filtered


def apply_mapping(mapping_df, variants_df):
    """Apply NM to NP mapping to variants DataFrame."""

    df_filtered = variants_df.join(
        mapping_df,
        mapping_df["NM_transcript"] == variants_df["transcription"],
        how="left",
    )
    return df_filtered.filter(F.col("NP_transcript").isNotNull())


def parse_fasta_sequences(fasta_file: str, prefix_filter: str) -> list:
    """
    Parse FASTA file and extract sequences with specified prefix.
    Args:
        fasta_file: Path to the FASTA file
        prefix_filter: Prefix to filter sequence IDs
    Returns:
        List of tuples (id, sequence) for filtered sequences"""
    records = [
        (record.id, str(record.seq))
        for record in SeqIO.parse(fasta_file, "fasta")
        if record.id.startswith(prefix_filter)
    ]
    return records


def process_protein_sequences(
    fasta_file: str,
    output_file: str,
    prefix_filter: str,
):
    """
    Process protein sequences from FASTA file and save to Parquet.
    """
    print(f"Processing protein sequences from {fasta_file}")
    records = parse_fasta_sequences(fasta_file, prefix_filter)

    if not records:
        print("No sequences to process")
        return
    df = spark.createDataFrame(records, ["id", "sequence"])
    df.write.mode("overwrite").parquet(output_file)

    print(f"Saved {len(records)} protein sequences to {output_file}")
    return df


def get_final_variation_dataset(
    mapped_df, output_file: str, join_sequences: bool = False, seq_file: str = None
):
    """
    Create final variation dataset with cleaned column names and optional sequence enrichment.

    Args:
        mapped_df: DataFrame with mapped variants (contains VariationID, Name, GeneSymbol, etc.)
        output_file: Path to save the final dataset
        join_sequences: Whether to join protein sequences and create mutated sequences
        seq_file: Path to protein sequences parquet file (required if join_sequences=True)

    Returns:
        DataFrame with final variation dataset

    Output Schema:
        - name (string): Variant name from ClinVar
        - gene_symbol (string): Gene symbol
        - phenotype_ids (string): Associated phenotype IDs
        - phenotype_list (string): List of phenotypes separated by |
        - NM_transcript (string): RefSeq mRNA accession (e.g., NM_000546)
        - NP_transcript (string): RefSeq protein accession (e.g., NP_000537)
        - REF_amino_acid (string): Reference amino acid (single letter code)
        - ALT_amino_acid (string): Alternate amino acid (single letter code)
        - position (string): Position in protein sequence
        - is_pathogenic (int): Clinical significance (1=pathogenic, 0=benign)
        - sequence (string): Original protein sequence (if join_sequences=True)
        - mutated_sequence (string): Mutated protein sequence (if join_sequences=True)
    """
    df = (
        mapped_df.filter(
            (F.col("VariationID").isNotNull()) & (F.col("VariationID") != "") & (F.col("SNV") != "")
        )
        .select(
            F.col("Name").alias("name"),
            F.col("GeneSymbol").alias("gene_symbol"),
            F.col("PhenotypeIDS").alias("phenotype_ids"),
            F.col("PhenotypeList").alias("phenotype_list"),
            F.col("transcription").alias("NM_transcript"),
            F.col("NP_transcript"),
            F.col("REF").alias("REF_amino_acid"),
            F.col("ALT").alias("ALT_amino_acid"),
            F.col("POS").alias("position"),
            F.col("ClinSigSimple").alias("is_pathogenic"),
        )
        .replace(AMINO_ACIDS, subset=["REF_amino_acid", "ALT_amino_acid"])
    )
    df_enriched = df
    if join_sequences:
        df_sequences = spark.read.parquet(seq_file)
        df_enriched = df.join(
            df_sequences, df["NP_transcript"] == df_sequences["id"], how="left"
        ).where(F.col("sequence").isNotNull())

        df_enriched = df_enriched.withColumn(
            "mutated_sequence",
            F.concat(
                F.expr("substring(sequence, 1, POS-1)"),
                F.col("ALT"),
                F.expr("substring(sequence, POS+1, length(sequence))"),
            ),
        )
    df_enriched.write.mode("overwrite").parquet(output_file)
    print(f"Final dataset saved to {output_file}")
    return df_enriched


def prepare_data_for_ml(full_data_file: str):
    """
    Prepare data for machine learning by creating sparse mutation vectors and filtering phenotypes.
    """
    df = spark.read.parquet(full_data_file).select(
        "name", "NP_transcript", "position", "phenotype_list", "is_pathogenic"
    )

    df_exploded = df.withColumn(
        "phenotype_list", F.explode(F.split(F.col("phenotype_list"), r"\|"))
    )

    def to_sparse(pos):
        if pos is None:
            return Vectors.sparse(MAX_SEQUENCE_LENGTH, [], [])
        return Vectors.sparse(MAX_SEQUENCE_LENGTH, [int(pos) - 1], [1.0])

    sparse_udf = F.udf(to_sparse, VectorUDT())
    df_sparse = df_exploded.where(f"position <= {MAX_SEQUENCE_LENGTH}").withColumn(
        "mutation_vector", sparse_udf(F.col("position"))
    )
    df_filtered = df_sparse.filter(
        (F.col("phenotype_list").isNotNull())
        & (F.col("phenotype_list") != "not provided")
        & (F.col("phenotype_list") != "not specified")
    ).dropDuplicates(["NP_transcript", "position", "phenotype_list"])

    return df_filtered

    # train_df, test_df = df_filtered.randomSplit([0.8, 0.2], seed=42)
    # save_to_parquet(train_df, "train_data.parquet")
    # save_to_parquet(test_df, "test_data.parquet")


def train_test_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split DataFrame into training, validation, and testing sets.

    Args:
        df: Input DataFrame
        train_ratio: Proportion of data to use for training set (default: 0.7)
        val_ratio: Proportion of data to use for validation set (default: 0.15)
        test_ratio: Proportion of data to use for testing set (default: 0.15)
        seed: Random seed for reproducibility

    Returns:
        tuple: (train_df, val_df, test_df)
            - train_df: Training DataFrame
            - val_df: Validation DataFrame
            - test_df: Testing DataFrame

    Note:
        train_ratio + val_ratio + test_ratio should equal 1.0
    """
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.001:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    train_df, val_df, test_df = df.randomSplit(
        [train_ratio, val_ratio, test_ratio], seed=seed
    )
    return train_df, val_df, test_df


def create_phenotype_pathogenic_vectors(df):
    """
    Create pathogenicity vectors for each (NP_transcript, phenotype_list) pair.
    Args:
        df: Input DataFrame with columns NP_transcript, phenotype_list, position, is_pathogenic
    Returns:
        DataFrame with added Patho_Vector column
    """

    @F.pandas_udf("array<int>", F.PandasUDFType.SCALAR)
    def pathogenic_vector(
        all_protein,
        all_phenotype,
        all_pos,
        all_is_pathogenic,
    ):
        vectors = []
        n = len(all_protein)

        df_all = pd.DataFrame(
            {
                "protein": all_protein,
                "phenotype": all_phenotype,
                "pos": all_pos,
                "is_pathogenic": all_is_pathogenic,
            }
        )

        for i in range(n):
            mask = (
                (df_all["protein"] == all_protein[i])
                & (df_all["phenotype"] == all_phenotype[i])
                & (df_all["is_pathogenic"] == 1)
                & (df_all.index != i)
            )
            positions = df_all.loc[mask, "pos"].tolist()
            vec = np.zeros(MAX_SEQUENCE_LENGTH, dtype=int)
            for p in positions:
                vec[int(p) - 1] = 1
            vectors.append(vec.tolist())
        return pd.Series(vectors)

    df = df.withColumn("is_pathogenic", F.col("is_pathogenic").cast(IntegerType()))

    df_final = df.withColumn(
        "context_vector",
        pathogenic_vector(
            df["NP_transcript"],
            df["phenotype_list"],
            df["position"],
            df["is_pathogenic"],
        ),
    )

    df_unique = df_final.dropDuplicates(["NP_transcript", "position", "phenotype_list"])
    return df_unique


def finalize_ml_datasets(df_train, df_val, df_test):
    def vector_to_list(v):
        if isinstance(v, SparseVector) or isinstance(v, DenseVector):
            return v.toArray().tolist()
        else:
            return None

    vector_to_list_udf = F.udf(vector_to_list, ArrayType(FloatType()))

    df_train = df_train.withColumn(
        "mutation_vector", vector_to_list_udf("mutation_vector")
    )
    df_val = df_val.withColumn("mutation_vector", vector_to_list_udf("mutation_vector"))
    df_test = df_test.withColumn(
        "mutation_vector", vector_to_list_udf("mutation_vector")
    )

    df_train.write.parquet("final_train.parquet")
    df_val.write.parquet("final_val.parquet")
    df_test.write.parquet("final_test.parquet")

    # train_masks = df_train.select("NP_transcript", "phenotype_list", "context_vector").dropDuplicates(
    #     ["NP_transcript", "phenotype_list"]
    # )
    # test_with_vectors = df_test.join(
    #     train_masks, on=["NP_transcript", "phenotype_list"], how="left"
    # ).withColumn(
    #     "context_vector",
    #     F.when(F.col("context_vector").isNull(), F.array([F.lit(0)] * 2056)).otherwise(
    #         F.col("context_vector")
    #     ),
    # )


def prepare_ml_dataset(path: str):
    df = prepare_data_for_ml(path)
    train_df, val_df, test_df = train_test_split(df)
    train_df = create_phenotype_pathogenic_vectors(train_df)
    val_df = create_phenotype_pathogenic_vectors(val_df)
    test_df = create_phenotype_pathogenic_vectors(test_df)
    finalize_ml_datasets(train_df, val_df, test_df)

def download_data():
    download_from_clinvar_ftp("pub/clinvar/tab_delimited/variant_summary.txt.gz", output_dir=".")
    download_from_clinvar_ftp("pub/clinvar/vcf_GRCh38/clinvar.vcf.gz", output_dir=".")
    download_from_clinvar_ftp("gene/DATA/gene2accession.gz", output_dir=".")
    download_from_clinvar_ftp("genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_protein.faa.gz", output_dir=".")

if __name__ == "__main__":
    args = parse_arguments()

    if args.join_sequences and not args.sequences_file:
        raise ValueError("--sequences-file is required when --join-sequences is set")

    try:
        if args.download_data:
            download_data()
        spark = setup_spark_environment(args.venv_python)

        variant_summary_df = process_variant_summary_file("variant_summary.txt")
        print(variant_summary_df.columns)
        vcf_df = process_vcf_file("clinvar.vcf", "clinvar_missense_ids.parquet")

        filtered_variants_df = variant_summary_df.join(
            vcf_df, on="VariationID", how="inner"
        )
        print(filtered_variants_df.columns)
        mapping_df = process_gene_mappings("gene2accession", "clinvar_mapping.parquet")

        mapped_df = apply_mapping(
            mapping_df=mapping_df, variants_df=filtered_variants_df
        )
        print(mapped_df.columns)

        process_protein_sequences(
            fasta_file="GCF_000001405.40_GRCh38.p14_protein.faa",
            output_file="output_sequences.parquet",
            prefix_filter="NP_",
        )

        get_final_variation_dataset(
            mapped_df=mapped_df,
            output_file="full_variation_dataset.parquet",
            join_sequences=args.join_sequences,
            seq_file=args.sequences_file if args.join_sequences else None,
        )

        if args.prepare_ml_dataset:
            print("Preparing ML dataset...")
            prepare_ml_dataset("full_variation_dataset.parquet")
            print("ML dataset preparation completed!")

        print("Pipeline completed successfully!")

    finally:
        spark.stop()
