import os
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf, regexp_extract, pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType, IntegerType
from pyspark.ml.linalg import Vectors, VectorUDT, SparseVector, DenseVector
from Bio import SeqIO
from pyspark.sql import SparkSession
import pysam
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

AMINO_ACIDS = {"Ala": "A", "Ile": "I", "Arg": "R","Leu": "L","Asn": "N","Lys": "K","Asp": "D","Met": "M","Asx": "B","Phe": "F","Cys":"C","Pro":"P","Gln":"Q","Ser":"S","Glu":"E","Thr":"T","Glx":"Z","Trp":"W","Gly":"G","Tyr":"Y","His":"H","Val":"V"}

def setup_spark_environment(venv_python_path: str = "/Users/kraszor/.venvs/clinvar_test/bin/python3.10") -> SparkSession:
    os.environ["PYSPARK_PYTHON"] = venv_python_path
    os.environ["PYSPARK_DRIVER_PYTHON"] = venv_python_path

    spark = SparkSession.builder \
        .appName("UniProt FASTA to Parquet with Biopython") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()
    
    return spark

spark = setup_spark_environment()

def read_variant_data(spark: SparkSession, file_path: str = "variant_summary.txt"):
    return spark.read.option("header", True).option("sep", "\t").csv(file_path)


def filter_clinvar_variants(df):
    return df.filter(
        (col("Type") == "single nucleotide variant") & 
        (col("Assembly") == "GRCh38") & 
        (col("ReviewStatus").isin([
            "criteria provided, single submitter", 
            "reviewed by expert panel", 
            "practice guideline"
        ])) & 
        (col("ClinicalSignificance").isin([
            "Pathogenic", 
            "Likely pathogenic", 
            "Benign", 
            "Likely benign"
        ]))
    )

def extract_variant_features(df):
    columns = ["Name", "VariationID", "GeneSymbol", "ClinSigSimple", 
               "PhenotypeIDS", "PhenotypeList", "ReviewStatus"]
    
    return df.select(*columns).withColumn(
        "transcription",
        regexp_extract(col("Name"), r"^([^()]+)\(", 1)
    ).withColumn(
        "gene",
        regexp_extract(col("Name"), r"\(([^)]+)\)", 1)
    ).withColumn(
        "SNV",
        regexp_extract(col("Name"), r"p\.([A-Za-z0-9]+)", 1)
    ).withColumn(
        "REF", 
        regexp_extract(col("SNV"), r"^([A-Za-z]+)", 1)
    ).withColumn(
        "POS", 
        regexp_extract(col("SNV"), r"([0-9]+)", 1)
    ).withColumn(
        "ALT", 
        regexp_extract(col("SNV"), r"([A-Za-z]+)$", 1)
    )


def save_to_parquet(df, output_path: str = "clinvar_snvs_filtered.parquet"):
    df.write.mode("overwrite").parquet(output_path)


def process_clinvar_data(input_file: str = "variant_summary.txt", 
                        output_file: str = "clinvar_snvs_filtered.parquet"):
        df = read_variant_data(spark, input_file)        
        df_filtered = filter_clinvar_variants(df)        
        df_processed = extract_variant_features(df_filtered)        
        save_to_parquet(df_processed, output_file)


def extract_missense_variants_from_vcf(vcf_file: str = "clinvar.vcf") -> list:
    missense_ids = []
    
    try:
        vcf_in = pysam.VariantFile(vcf_file)
        for rec in vcf_in:
            mc = rec.info.get("MC")
            if mc and any("missense_variant" in m for m in mc):
                missense_ids.append(rec.id)
        vcf_in.close()
    except Exception as e:
        raise
    
    return missense_ids


def save_missense_ids_to_parquet(missense_ids: list, output_file: str = "clinvar_missense_ids.parquet"):
    if not missense_ids:
        print("Brak wariantów missense do zapisania")
        return
    
    df = pd.DataFrame({"VariationID": missense_ids})
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)
    
    print(f"Zapisano {len(missense_ids)} wariantów missense do {output_file}")


def process_missense_variants(vcf_file: str = "clinvar.vcf", 
                            output_file: str = "clinvar_missense_ids.parquet"):
    print(f"Przetwarzanie wariantów missense z pliku: {vcf_file}")
    missense_ids = extract_missense_variants_from_vcf(vcf_file)
    
    print(f"Liczba wariantów missense: {len(missense_ids)}")
    save_missense_ids_to_parquet(missense_ids, output_file)


def merge_missense_with_filtered_data(missense_ids_file: str = "clinvar_missense_ids.parquet",
                                    filtered_data_file: str = "clinvar_snvs_filtered.parquet",
                                    output_file: str = "clinvar_missense_filtered.parquet"):
    print(f"Łączenie danych z {missense_ids_file} i {filtered_data_file}")
    
    try:
        df_ids = spark.read.parquet(missense_ids_file)
        df_full = spark.read.parquet(filtered_data_file)
        df_filtered = df_full.join(df_ids, on="VariationID", how="inner")
        save_to_parquet(df_filtered, output_file)
    except Exception as e:
        print(f"Błąd podczas łączenia danych: {e}")
        raise


def read_gene_accession_data(file_path: str = "../gene2accession"):
    return spark.read.option("header", True).option("sep", "\t").csv(file_path)


def filter_reviewed_gene_mappings(df):
    return df.filter(
        col("status").isin(["REVIEWED", "VALIDATED"])
    ).select(
        col("`RNA_nucleotide_accession.version`").alias("NM"), 
        col("`protein_accession.version`").alias("NP")
    ).where(
        (col("NM") != "-") & (col("NP") != "-")
    ).distinct()


def process_gene_mappings(input_file: str = "../gene2accession",
                         output_file: str = "clinvar_mapping.parquet"):

    df = read_gene_accession_data(input_file)
    df_filtered = filter_reviewed_gene_mappings(df)
    save_to_parquet(df_filtered, output_file)


def create_final_dataset(mapping_file: str = "clinvar_mapping.parquet",
                        missense_file: str = "clinvar_missense_filtered.parquet",
                        output_file: str = "final_data.parquet"):
    
    try:
        df_mapping = spark.read.parquet(mapping_file)
        df_full = spark.read.parquet(missense_file)
        
        df_filtered = df_full.join(
            df_mapping, 
            df_mapping["NM"] == df_full["transcription"], 
            how="left"
        )
        
        save_to_parquet(df_filtered, output_file)        
        count = df_filtered.count()
        count_with_mapping = df_filtered.filter(col("NM").isNotNull()).count()
        print(f"Zapisano {count} rekordów do {output_file}")
        print(f"Z mapowaniem NM-NP: {count_with_mapping} rekordów")
        
    except Exception as e:
        print(f"Błąd podczas tworzenia finalnego zbioru danych: {e}")
        raise


def parse_fasta_sequences(fasta_file: str, prefix_filter: str = "NP_") -> list:
    try:
        records = [
            (record.id, str(record.seq))
            for record in SeqIO.parse(fasta_file, "fasta")
            if record.id.startswith(prefix_filter)
        ]
        return records
        
    except Exception as e:
        print(f"Błąd podczas parsowania pliku FASTA: {e}")
        raise


def process_protein_sequences(fasta_file: str = "GCF_000001405.40_GRCh38.p14_protein.faa",
                            output_file: str = "output_sequences.parquet",
                            prefix_filter: str = "NP_"):
    print(f"Przetwarzanie sekwencji białek z {fasta_file}")
    records = parse_fasta_sequences(fasta_file, prefix_filter)
    
    if not records:
        print("Brak sekwencji do przetworzenia")
        return
    df = spark.createDataFrame(records, ["id", "sequence"])
    save_to_parquet(df, output_file)
    
    print(f"Zapisano {len(records)} sekwencji białek do {output_file}")


def enrich_with_protein_sequences(final_data_file: str = "final_data.parquet",
                                sequences_file: str = "output_sequences.parquet",
                                output_file: str = "final_data_seq.parquet"):
    
    try:
        df_full = spark.read.parquet(final_data_file)
        df_sequences = spark.read.parquet(sequences_file)
        
        df_enriched = df_full.join(
            df_sequences, 
            df_full["NP"] == df_sequences["id"], 
            how="left"
        )        
        save_to_parquet(df_enriched, output_file)
        
        total_count = df_enriched.count()
        with_sequence_count = df_enriched.filter(col("sequence").isNotNull()).count()
        
        print(f"Zapisano {total_count} rekordów do {output_file}")
        print(f"Z sekwencjami białek: {with_sequence_count} rekordów")
        print(f"Pokrycie sekwencjami: {(with_sequence_count/total_count*100):.1f}%")
        
    except Exception as e:
        print(f"Błąd podczas wzbogacania o sekwencje: {e}")
        raise


def create_mutated_sequences_dataset(input_file: str = "final_data_seq.parquet",
                                   output_file: str = "data_with_modified_seq.parquet"):
    df = spark.read.parquet(input_file).where(
        F.col("id").isNotNull()
    ).where(
        F.col("id") != ""
    ).where(
        F.col("SNV") != ""
    ).select(
        "Name", "GeneSymbol", "PhenotypeIDS", "PhenotypeList", 
        "transcription", "NP", "REF", "ALT", "POS", "sequence", "ClinSigSimple"
    ).replace(AMINO_ACIDS, subset=["REF", "ALT"]).withColumn(
        "mutated_sequence",
        F.concat(
            F.expr("substring(sequence, 1, POS-1)"),
            F.col("ALT"),
            F.expr("substring(sequence, POS+1, length(sequence))")
        )
    )
    save_to_parquet(df, output_file)


def prepare_data_for_ml():
    df = spark.read.parquet("data_with_modified_seq.parquet").select("Name", "NP","POS", "PhenotypeList", "ClinSigSimple")
    df_exploded = df.withColumn("PhenotypeList", F.explode(F.split(F.col("PhenotypeList"), r"\|")))
    
    def to_sparse(pos):
        if pos is None:
            return Vectors.sparse(2056, [], [])
        return Vectors.sparse(2056, [int(pos)-1], [1.0])
    
    sparse_udf = F.udf(to_sparse, VectorUDT())
    df_sparse = df_exploded.where("POS <= 2056").withColumn("POS_vector", sparse_udf(F.col("POS")))
    df_filtered = df_sparse.filter(
        (F.col("PhenotypeList").isNotNull()) & 
        (F.col("PhenotypeList") != "not provided") & 
        (F.col("PhenotypeList") != "not specified")
    ).dropDuplicates(["NP", "POS", "PhenotypeList"])
    
    train_df, test_df = df_filtered.randomSplit([0.8, 0.2], seed=42)
    save_to_parquet(train_df, "train_data.parquet")
    save_to_parquet(test_df, "test_data.parquet")


def create_pathogenic_vectors():
    train_df = spark.read.parquet("train_data.parquet")
    
    max_len = 2056
    
    @pandas_udf("array<int>", PandasUDFType.SCALAR)
    def pathogenic_vector(protein, phenotype, pos, clinsig, all_protein, all_phenotype, all_pos, all_clinsig):
        vectors = []
        n = len(protein)
        
        df_all = pd.DataFrame({
            "Protein": all_protein,
            "Phenotype": all_phenotype,
            "POS": all_pos,
            "ClinSigSimple": all_clinsig
        })
        
        for i in range(n):
            mask = (
                (df_all["Protein"] == protein[i]) &
                (df_all["Phenotype"] == phenotype[i]) &
                (df_all["ClinSigSimple"] == 1) &
                (df_all.index != i)
            )
            positions = df_all.loc[mask, "POS"].tolist()
            vec = np.zeros(max_len, dtype=int)
            for p in positions:
                vec[int(p)-1] = 1
            vectors.append(vec.tolist())
        return pd.Series(vectors)
    
    train_df = train_df.withColumn("ClinSigSimple", F.col("ClinSigSimple").cast(IntegerType()))
    
    df_final = train_df.withColumn(
        "Patho_Vector",
        pathogenic_vector(
            train_df["NP"],
            train_df["PhenotypeList"],
            train_df["POS"],
            train_df["ClinSigSimple"],
            train_df["NP"],
            train_df["PhenotypeList"],
            train_df["POS"],
            train_df["ClinSigSimple"]
        )
    )
    
    df_train_unique = df_final.dropDuplicates(["NP", "POS", "PhenotypeList"])
    save_to_parquet(df_train_unique, "train_with_vectors.parquet")


def finalize_ml_datasets():
    df_train = spark.read.parquet("train_with_vectors.parquet")
    df_test = spark.read.parquet("test_data.parquet").withColumn("ClinSigSimple", F.col("ClinSigSimple").cast(IntegerType()))
    
    def vector_to_list(v):
        if isinstance(v, SparseVector) or isinstance(v, DenseVector):
            return v.toArray().tolist()
        else:
            return None
    
    vector_to_list_udf = udf(vector_to_list, ArrayType(FloatType()))
    
    df_train = df_train.withColumn("POS_vector", vector_to_list_udf("POS_vector"))
    df_test = df_test.withColumn("POS_vector", vector_to_list_udf("POS_vector"))
    
    save_to_parquet(df_train, "final_train.parquet")
    save_to_parquet(df_test, "final_test.parquet")
    
    train_masks = df_train.select("NP", "PhenotypeList", "Patho_Vector").dropDuplicates(["NP", "PhenotypeList"])
    test_with_vectors = (
        df_test
        .join(train_masks, on=["NP", "PhenotypeList"], how="left")
        .withColumn(
            "Patho_Vector",
            F.when(F.col("Patho_Vector").isNull(), F.array([F.lit(0)] * 2056)).otherwise(F.col("Patho_Vector"))
        )
    )
    
    save_to_parquet(test_with_vectors, "final_test_with_vectors.parquet")


def prepare_ml_dataset():
    prepare_data_for_ml()
    create_pathogenic_vectors()
    finalize_ml_datasets()


if __name__ == "__main__":
    try:
        process_clinvar_data()
        process_missense_variants()
        merge_missense_with_filtered_data()
        process_gene_mappings()
        create_final_dataset()
        process_protein_sequences()
        enrich_with_protein_sequences()
        create_mutated_sequences_dataset()
        prepare_ml_dataset()
        
    finally:
        spark.stop()