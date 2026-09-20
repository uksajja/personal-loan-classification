
from pathlib import Path
import urllib.request
import zipfile

import pandas as pd

# -----------------------------------------------------------------------------
# Data ingestion helpers
# -----------------------------------------------------------------------------

class DataIngestor:
    """A helper class to perform data ingestion and CSV loading."""

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------
    def __init__(self, config: dict | None = None):
        """Initialize ingestion with an existing DataFrame or a CSV path."""
        if config is None:
            raise ValueError("Config dictionary must be provided.")
        self.config = config
        self.original_df = None
    
    def load_data(self, **kwargs) -> pd.DataFrame:
        data_source_type = self.config.get("data_source_type", "local_data_file")   
        if data_source_type == "local_data_file":
            local_data_file = self.config.get("local_data_file")
            if not local_data_file:
                raise ValueError("Local data file path is required for 'local_data_file' data source type.")
            df = self.load_data_from_file(local_data_file)
        elif data_source_type == "remote_data_file":
            source_URL = self.config.get("source_URL")
            if not source_URL:
                raise ValueError("Source URL is required for 'remote_data_file' data source type.")
            dest_path = self.config.get("dest_path")
            df = self.load_data_from_url(source_URL, dest_path)
        else:
            raise ValueError(f"Unsupported data source type: {data_source_type}")

        return df

    # ---------------------------------------------------------------------
    # CSV loading utilities
    # ---------------------------------------------------------------------
    def load_data_from_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load dataset from a local file and keep an internal copy.
        
        Supports multiple file formats: CSV, Parquet, Excel (xlsx/xls), 
        JSON, HDF5, Feather, ORC, and others supported by pandas.
        Also supports ZIP files - will extract and load the first data file found.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments to pass to the pandas read function
                     (e.g., sheet_name for Excel, compression for CSV)
        
        Returns:
            pd.DataFrame: The loaded dataframe
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        # Handle ZIP files - extract and load the first data file found
        if suffix == ".zip":
            extract_dir = file_path.parent / file_path.stem
            self.unzip_file(file_path, extract_dir)
            
            # Supported data file extensions
            supported_extensions = {
                ".csv", ".parquet", ".pq", ".xlsx", ".xls", ".xlsm",
                ".json", ".h5", ".hdf", ".hdf5"
            }
            
            # Find the first data file with a supported extension
            data_files = [
                f for f in extract_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in supported_extensions
            ]
            
            if not data_files:
                raise ValueError(
                    f"No supported data files found in {file_path}. "
                    f"Supported formats: {', '.join(supported_extensions)}"
                )
            
            # Use the first data file found
            selected_file = data_files[0]
            if len(data_files) > 1:
                print(f"Multiple data files found. Loading: {selected_file.name}")
            
            return self.load_data_from_file(str(selected_file), **kwargs)
        
        # Process non-zip files based on extension
        
        # CSV files
        if suffix == ".csv":
            df = pd.read_csv(file_path, **kwargs)
        
        # Parquet files
        elif suffix == ".parquet" or suffix == ".pq":
            self.df = pd.read_parquet(file_path, **kwargs)
        
        # Excel files
        elif suffix in [".xlsx", ".xls", ".xlsm"]:
            df = pd.read_excel(file_path, **kwargs)
        
        # JSON files
        elif suffix == ".json":
            df = pd.read_json(file_path, **kwargs)
        
        # HDF5 files
        elif suffix in [".h5", ".hdf", ".hdf5"]:
            # For HDF5, key parameter is required; use default if not provided
            key = kwargs.pop("key", "data")
            df = pd.read_hdf(file_path, key, **kwargs)
        
        # SQL and other formats via SQL connection
        elif suffix in [".db", ".sqlite", ".sqlite3"]:
            raise NotImplementedError("SQL database loading requires additional setup. Use read_sql() directly.")
        
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. Supported formats: "
                ".csv, .parquet, .xlsx, .xls, .xlsm, .json, .h5, .hdf, .hdf5"
            )
        
        self.original_df = df.copy()
        return df

    def load_data_from_url(self, url: str, dest_path: str | Path | None = None) -> pd.DataFrame:
        """Download a CSV from a URL and load it into the ingestion object."""
        dest_path = Path(dest_path) if dest_path is not None else Path(url.split("/")[-1])
        self.download_file(url, dest_path)
        return self.load_data(dest_path)

    # ---------------------------------------------------------------------
    # File download and extraction helpers
    # ---------------------------------------------------------------------
    def download_file(self, url: str, dest_path: str | Path) -> Path:
        """Download a file from a URL to a local destination."""
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest_path)
        return dest_path

    def unzip_file(self, zip_path: str | Path, extract_to: str | Path | None = None) -> Path:
        """Extract a ZIP archive to a local directory."""
        zip_path = Path(zip_path)
        extract_to = Path(extract_to) if extract_to is not None else zip_path.with_suffix("")
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        return extract_to
    
    def _ensure_data_loaded(self) -> None:
        if self.df is None:
            raise ValueError("No DataFrame is loaded. Call load_data() first or provide a DataFrame at initialization.")
        else:
            print(f"Data loaded successfully with shape: {self.df.shape}")
    
    def reset(self) -> pd.DataFrame:
        """Reset the working DataFrame to the original loaded data."""
        if self.original_df is None:
            raise ValueError("No original DataFrame is available to reset.")
        df = self.original_df.copy()
        return df
