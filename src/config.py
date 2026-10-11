"""
Configuration module for TuneTrail: Spotify Playlist Association Rule Mining.
Centralizes paths, parameters, and tuning constants.
"""

from pathlib import Path

# Resolve project root: works whether invoked from tunetrail/ or workspace root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report"

# Ensure runtime directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Processed file paths
BASKET_PARQUET = PROCESSED_DATA_DIR / "basket.parquet"
BASKET_SPARSE_NPZ = PROCESSED_DATA_DIR / "basket_sparse.npz"
BASKET_COLUMNS_JSON = PROCESSED_DATA_DIR / "basket_columns.json"
ITEM_META_PARQUET = PROCESSED_DATA_DIR / "item_meta.parquet"
TRANSACTIONS_PARQUET = PROCESSED_DATA_DIR / "transactions.parquet"
PREPROCESS_STATS_JSON = PROCESSED_DATA_DIR / "preprocess_stats.json"
SWEEP_RESULTS_CSV = RESULTS_DIR / "sweep_results.csv"

# Preprocessing Tuning Constants
SAMPLE_PCT = 20                 # Default 20% user sampling for memory safety; set to 100 for full dataset
MIN_SUPPORT_FLOOR = 0.0005      # 0.05% of playlists minimum frequency for item universe
MAX_ITEMS = 20000               # Upper bound on item universe size
MIN_PLAYLIST_LEN = 2            # Playlists with < 2 items cannot form pairs
LENGTH_CAP_PERCENTILE = 99      # Playlist length cutoff percentile (typically 99th percentile)

# Mining Defaults
DEFAULT_MIN_SUPPORT = 0.002     # 0.2% support default
DEFAULT_MIN_CONFIDENCE = 0.25   # 25% confidence default
DEFAULT_MIN_LIFT = 1.5          # 1.5x lift default
DEFAULT_MAX_LEN = 3             # 3 items max per frequent itemset
DEFAULT_MIN_COOCCURRENCE = 5    # Minimum co-occurrence playlist count
DEFAULT_ALGORITHM = "fpgrowth"  # Default algorithm: 'fpgrowth' or 'apriori'


def find_raw_dataset_file():
    """
    Locates the raw Kaggle dataset in RAW_DATA_DIR.
    Checks for .csv files first, then .zip files.
    Also checks parent and sister directories if needed.
    """
    # 1. Search in RAW_DATA_DIR: prioritize non-synthetic CSV files
    csvs = list(RAW_DATA_DIR.glob("**/*.csv"))
    real_csvs = [c for c in csvs if "synthetic" not in c.name.lower()]
    if real_csvs:
        for c in real_csvs:
            if "spotify" in c.name.lower() or "playlist" in c.name.lower():
                return c
        return real_csvs[0]

    # Fallback to any other csv if real not found
    if csvs:
        return csvs[0]

    zips = list(RAW_DATA_DIR.glob("**/*.zip"))
    if zips:
        return zips[0]

    # 2. Search in workspace root and parent Downloads folder
    workspace_dir = PROJECT_ROOT.parent
    for search_dir in [workspace_dir, workspace_dir.parent]:
        try:
            cand_csvs = list(search_dir.glob("*.csv"))
            for c in cand_csvs:
                if "spotify" in c.name.lower() or "playlist" in c.name.lower():
                    return c
            cand_zips = list(search_dir.glob("*.zip"))
            for z in cand_zips:
                if "spotify" in z.name.lower() or "larxel" in z.name.lower() or "playlist" in z.name.lower():
                    return z
        except Exception:
            pass

    return None


def get_dataset_stream(raw_file_path):
    """
    Returns a Path to a readable CSV.
    If the file is a zip archive, extracts the inner CSV to RAW_DATA_DIR.
    """
    import zipfile
    path = Path(raw_file_path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, 'r') as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
            if not csv_names:
                raise ValueError(f"No CSV file found inside {path}")
            csv_to_extract = csv_names[0]
            print(f"[data] Found '{csv_to_extract}' in zip archive. Extracting to {RAW_DATA_DIR}...")
            z.extract(csv_to_extract, path=RAW_DATA_DIR)
            return RAW_DATA_DIR / csv_to_extract
    return path

