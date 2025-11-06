import os
import gzip
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Optional
from pyspark.sql import SparkSession

def setup_spark_environment(venv_python_path: str) -> SparkSession:
    """
    Set up Spark environment with specified Python interpreter from virtual environment.
    """
    os.environ["PYSPARK_PYTHON"] = venv_python_path
    os.environ["PYSPARK_DRIVER_PYTHON"] = venv_python_path

    spark = SparkSession.builder \
        .appName("BIO data processing") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()
    
    return spark


def unpack_gz_file(gz_file_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Unpack a .gz file.
    
    Args:
        gz_file_path: Path to the .gz file
        output_path: Path for the unpacked file (if None, removes .gz extension)
    
    Returns:
        Path to unpacked file if successful, None if failed
    """
    gz_path = Path(gz_file_path)
    
    if not gz_path.exists():
        print(f"File not found: {gz_file_path}")
        return None
    
    if output_path is None:
        # Remove .gz extension
        if gz_path.suffix == ".gz":
            output_path = str(gz_path.with_suffix(""))
        else:
            output_path = str(gz_path) + ".unpacked"
    
    try:
        print(f"Unpacking {gz_path.name}...")
        with gzip.open(gz_file_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"Successfully unpacked to: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error unpacking {gz_file_path}: {e}")
        return None


def download_from_clinvar_ftp(
    file_path: str,
    output_dir: str = "./data",
    base_url: str = "https://ftp.ncbi.nlm.nih.gov/",
    unpack: bool = True
) -> Optional[str]:
    """
    Download a file from ClinVar FTP using curl and optionally unpack .gz files.
    
    Args:
        file_path: Relative path to the file on ClinVar FTP (e.g., "vcf_GRCh38/clinvar.vcf.gz")
        output_dir: Local directory to save the file
        base_url: Base URL for ClinVar FTP
        unpack: Whether to unpack .gz files after download
    
    Returns:
        Path to downloaded (and unpacked if requested) file if successful, None if failed
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    full_url = f"{base_url.rstrip('/')}/{file_path.lstrip('/')}"
    filename = Path(file_path).name
    output_path = Path(output_dir) / filename
    
    try:
        cmd = [
            "curl",
            "-L",
            "-f",
            "-o", str(output_path),
            "--progress-bar",
            full_url
        ]
        
        print(f"Downloading {filename} from NCBI FTP...")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"Successfully downloaded: {output_path}")
            
            # Unpack if requested and file is .gz
            if unpack and str(output_path).endswith('.gz'):
                unpacked_path = unpack_gz_file(str(output_path))
                if unpacked_path:
                    return unpacked_path
                else:
                    print("Unpacking failed, returning compressed file path")
                    return str(output_path)
            
            return str(output_path)
        else:
            print("Download failed: File not created or empty")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {filename}: {e}")
        print(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error downloading {filename}: {e}")
        return None


def parse_arguments():
    """
    Parse command line arguments for data processing pipeline.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="ClinVar data processing pipeline for variant analysis"
    )
    
    parser.add_argument(
        "--venv-python",
        required=True,
        type=str,
        help="Path to Python executable in virtual environment (e.g., /path/to/venv/bin/python3.10)"
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Whether to download necessary data files from ClinVar FTP"
    )
    
    parser.add_argument(
        "--join-sequences",
        action="store_true",
        help="Whether to join protein sequences and create mutated sequences"
    )
    
    parser.add_argument(
        "--sequences-file",
        type=str,
        default="output_sequences.parquet",
        help="Path to protein sequences parquet file (required if --join-sequences is set)"
    )
    
    parser.add_argument(
        "--prepare-ml-dataset",
        action="store_true",
        help="Whether to prepare machine learning dataset with train/val/test splits"
    )
    
    return parser.parse_args()