"""
preprocess.py: Step 2 - Data Preprocessing & Transaction Matrix Construction.

Implements the exact cleaning pipeline:
1. Missing value removal (artist, track, playlistname, user_id)
2. Text normalization & suffix removal (Remastered, Live, Radio Edit, etc.)
3. Transaction identification: tid = user_id + "|" + playlistname
4. Intra-playlist track de-duplication: (tid, item)
5. Playlist length filtering: min 2 items, max at 99th percentile cap
6. User-level duplicate playlist removal: deduplicate on set of items per user
7. Item universe filtering based on minimum support floor and top 20k cap
8. Re-filtering playlists < 2 items
9. Sparse CSR matrix and basket DataFrame construction
10. Item metadata table and statistics logging (preprocess_stats.json)
"""

import re
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    BASKET_PARQUET,
    BASKET_SPARSE_NPZ,
    BASKET_COLUMNS_JSON,
    ITEM_META_PARQUET,
    TRANSACTIONS_PARQUET,
    PREPROCESS_STATS_JSON,
    SAMPLE_PCT,
    MIN_SUPPORT_FLOOR,
    MAX_ITEMS,
    MIN_PLAYLIST_LEN,
    LENGTH_CAP_PERCENTILE,
    find_raw_dataset_file,
    get_dataset_stream
)

# Suffix cleaning regex
SUFFIX_PATTERN = re.compile(
    r"\s*[-–/]\s*("
    r"remaster(ed)?(\s*\d{4})?|"
    r"live(\s*at\s*[^-\)]+)?|"
    r"radio\s*edit|"
    r"single\s*version|"
    r"album\s*version|"
    r"deluxe(\s*edition)?|"
    r"bonus\s*track|"
    r"acoustic(\s*version)?|"
    r"mono|"
    r"stereo|"
    r"original\s*mix|"
    r"club\s*mix|"
    r"extended\s*mix|"
    r"\d{4}\s*remaster"
    r")\b.*$",
    flags=re.IGNORECASE
)

WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str, is_track: bool = False) -> str:
    """
    Normalizes text: lowercase, strip leading/trailing spaces, collapse multiple spaces,
    and strips common noisy edition/remaster suffixes for track titles.
    """
    if not isinstance(text, str):
        return ""
    cleaned = text.strip().lower()
    if is_track:
        cleaned = SUFFIX_PATTERN.sub("", cleaned).strip()
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned


def run_preprocessing(
    raw_path=None,
    sample_pct=SAMPLE_PCT,
    min_support_floor=MIN_SUPPORT_FLOOR,
    max_items=MAX_ITEMS,
    min_playlist_len=MIN_PLAYLIST_LEN,
    length_cap_percentile=LENGTH_CAP_PERCENTILE
):
    start_time = time.time()
    print("=" * 70)
    print("TUNETRAIL PREPROCESSING PIPELINE")
    print("=" * 70)

    if raw_path is None:
        raw_path = find_raw_dataset_file()

    if raw_path is None:
        raise FileNotFoundError(
            f"No raw dataset found in {RAW_DATA_DIR}. Please place CSV or ZIP file there."
        )

    csv_path = get_dataset_stream(raw_path)
    print(f"[1/8] Reading raw data from: {csv_path.name}")

    # Read CSV in chunks for memory safety
    required_cols = ["user_id", "artistname", "trackname", "playlistname"]
    raw_rows = 0
    dropped_na = 0
    chunks = []

    # If sampling users, pick a reproducible set of users from eda_summary or via hash
    eda_summary_path = PROCESSED_DATA_DIR / "eda_summary.json"
    sampled_users = None
    if sample_pct < 100:
        print(f"[Sampling] Target: Retaining {sample_pct}% of unique users...")

    reader = pd.read_csv(
        csv_path,
        quotechar='"',
        escapechar='\\',
        skipinitialspace=True,
        on_bad_lines="skip",
        chunksize=500_000,
        dtype=str
    )

    for chunk in reader:
        chunk.columns = [c.strip().strip('"').strip().lower() for c in chunk.columns]
        raw_rows += len(chunk)
        clean_chunk = chunk.dropna(subset=required_cols).copy()
        dropped_na += (len(chunk) - len(clean_chunk))

        if sample_pct < 100:
            # Deterministic user hash so all playlists of a sampled user are kept intact
            user_mask = clean_chunk["user_id"].apply(lambda u: (abs(hash(u)) % 100) < sample_pct)
            clean_chunk = clean_chunk[user_mask]

        if not clean_chunk.empty:
            chunks.append(clean_chunk)

    df = pd.concat(chunks, ignore_index=True)
    rows_after_dropna = len(df)
    print(f"      Loaded {raw_rows:,} raw rows ({rows_after_dropna:,} retained after cleaning/sampling).")
    print(f"      Dropped {dropped_na:,} rows with missing values.")

    # 2. Text Normalization
    print("[3/8] Normalizing text (lowercasing, trimming, suffix stripping)...")
    raw_unique_tracks = (df["artistname"] + " – " + df["trackname"]).nunique()
    df["artist_clean"] = df["artistname"].apply(lambda s: normalize_text(s, is_track=False))
    df["track_clean"] = df["trackname"].apply(lambda s: normalize_text(s, is_track=True))
    df["item"] = df["artist_clean"] + " – " + df["track_clean"]
    df["tid"] = df["user_id"] + "|" + df["playlistname"].str.strip()
    clean_unique_tracks = df["item"].nunique()
    tracks_merged = raw_unique_tracks - clean_unique_tracks
    print(f"      Unique tracks before normalization: {raw_unique_tracks:,}")
    print(f"      Unique tracks after normalization:  {clean_unique_tracks:,} ({tracks_merged:,} redundant variants merged)")

    # 3. Remove intra-playlist duplicates
    print("[4/8] Removing duplicate (tid, item) occurrences within playlists...")
    rows_before_dedup = len(df)
    df = df.drop_duplicates(subset=["tid", "item"]).copy()
    intra_playlist_dups_removed = rows_before_dedup - len(df)
    print(f"      Removed {intra_playlist_dups_removed:,} duplicate tracks inside playlists.")

    # 4. Playlist Length Filtering (short and long caps)
    print(f"[5/8] Filtering playlist lengths (min={min_playlist_len}, max={length_cap_percentile}th percentile)...")
    playlist_sizes = df.groupby("tid").size()
    total_initial_playlists = len(playlist_sizes)
    length_cap = int(np.percentile(playlist_sizes, length_cap_percentile))
    print(f"      Initial playlists: {total_initial_playlists:,}")
    print(f"      Calculated {length_cap_percentile}th percentile cap: {length_cap} tracks")

    valid_tids_length = playlist_sizes[(playlist_sizes >= min_playlist_len) & (playlist_sizes <= length_cap)].index
    df = df[df["tid"].isin(valid_tids_length)].copy()
    playlists_after_length_filter = df["tid"].nunique()
    print(f"      Playlists after length filter: {playlists_after_length_filter:,} ({total_initial_playlists - playlists_after_length_filter:,} removed)")

    # 5. Remove Duplicate Playlists per User
    print("[6/8] Removing duplicate playlists from the same user (identical item sets)...")
    # Group items per user playlist
    user_playlist_items = df.groupby(["user_id", "tid"])["item"].apply(lambda s: frozenset(s)).reset_index()
    before_user_dedup = len(user_playlist_items)
    unique_user_playlists = user_playlist_items.drop_duplicates(subset=["user_id", "item"])
    duplicate_playlists_removed = before_user_dedup - len(unique_user_playlists)
    valid_tids = set(unique_user_playlists["tid"])
    df = df[df["tid"].isin(valid_tids)].copy()
    print(f"      Removed {duplicate_playlists_removed:,} duplicate playlists from same users ({len(valid_tids):,} remaining).")

    # 6. Item Universe Filtering
    N_playlists = df["tid"].nunique()
    min_count_floor = max(2, int(np.ceil(min_support_floor * N_playlists)))
    print(f"[7/8] Filtering item universe (min_support={min_support_floor:.4%} => min {min_count_floor} playlists, top {max_items:,} cap)...")
    item_counts = df["item"].value_counts(sort=False)
    items_meeting_floor = item_counts[item_counts >= min_count_floor]
    print(f"      Items with >= {min_count_floor} occurrences: {len(items_meeting_floor):,} (out of {len(item_counts):,})")

    # Keep top max_items
    kept_items = set(items_meeting_floor.nlargest(max_items).index)
    df = df[df["item"].isin(kept_items)].copy()

    # Re-filter playlists that now have fewer than min_playlist_len
    final_playlist_sizes = df.groupby("tid").size()
    final_valid_tids = final_playlist_sizes[final_playlist_sizes >= min_playlist_len].index
    df = df[df["tid"].isin(final_valid_tids)].copy()
    final_N_playlists = df["tid"].nunique()
    final_N_items = df["item"].nunique()
    print(f"      Final valid playlists: {final_N_playlists:,}")
    print(f"      Final valid items:     {final_N_items:,}")
    print(f"      Final transaction rows: {len(df):,}")

    # 7. Build Sparse CSR Matrix & Item Metadata
    print("[8/8] Constructing sparse matrix and saving artifacts...")
    unique_items = sorted(df["item"].unique())
    item_to_idx = {item: i for i, item in enumerate(unique_items)}
    col_indices = df["item"].map(item_to_idx).to_numpy()
    row_indices, row_ids = pd.factorize(df["tid"])

    X = csr_matrix((np.ones(len(df), dtype=bool), (row_indices, col_indices)), shape=(len(row_ids), len(unique_items)))
    save_npz(BASKET_SPARSE_NPZ, X)

    # Save exact column names in order
    with open(BASKET_COLUMNS_JSON, "w", encoding="utf-8") as f:
        json.dump(unique_items, f, indent=2)

    # Save compact transactions parquet: tid, user_id, items_list in exact row order of X
    tid_df = df.groupby("tid").agg({
        "user_id": "first",
        "playlistname": "first",
        "item": list
    }).reindex(row_ids).reset_index()
    tid_df["track_count"] = tid_df["item"].apply(len)
    tid_df.to_parquet(TRANSACTIONS_PARQUET, index=False)

    # Item metadata table
    item_freq = df.groupby("item").agg({
        "artist_clean": "first",
        "track_clean": "first",
        "tid": "nunique"
    }).reset_index()
    item_freq.columns = ["item", "artist", "track", "playlist_count"]
    item_freq["support"] = item_freq["playlist_count"] / float(final_N_playlists)
    item_freq = item_freq.sort_values(by="playlist_count", ascending=False).reset_index(drop=True)
    item_freq.to_parquet(ITEM_META_PARQUET, index=False)

    # Build dense basket dataframe only if small enough, or sparse DataFrame
    try:
        # Save sparse dataframe to parquet
        sparse_df = pd.DataFrame.sparse.from_spmatrix(X, index=row_ids, columns=unique_items)
        sparse_df.to_parquet(BASKET_PARQUET)
        print(f"      Saved basket DataFrame to: {BASKET_PARQUET.name}")
    except Exception as e:
        print(f"      Saved sparse CSR to npz ({e}).")

    elapsed_time = time.time() - start_time

    # Preprocess statistics record for DATA_CARD and report
    stats = {
        "raw_file": str(csv_path.name),
        "raw_rows": int(raw_rows),
        "missing_values_dropped": int(dropped_na),
        "user_sample_pct": float(sample_pct),
        "raw_unique_tracks": int(raw_unique_tracks),
        "normalized_unique_tracks": int(clean_unique_tracks),
        "variant_tracks_merged": int(tracks_merged),
        "intra_playlist_duplicates_removed": int(intra_playlist_dups_removed),
        "initial_playlists": int(total_initial_playlists),
        "playlist_length_cap_pct": int(length_cap_percentile),
        "playlist_length_cap": int(length_cap),
        "playlists_after_length_filter": int(playlists_after_length_filter),
        "duplicate_user_playlists_removed": int(duplicate_playlists_removed),
        "min_support_floor": float(min_support_floor),
        "min_count_floor": int(min_count_floor),
        "max_items_cap": int(max_items),
        "final_playlists": int(final_N_playlists),
        "final_items": int(final_N_items),
        "final_rows": int(len(df)),
        "matrix_sparsity": float(1.0 - (len(df) / (final_N_playlists * final_N_items))),
        "preprocessing_time_sec": round(elapsed_time, 2)
    }

    with open(PREPROCESS_STATS_JSON, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print("=" * 70)
    print("PREPROCESSING COMPLETE!")
    print(f"Stats saved to: {PREPROCESS_STATS_JSON}")
    print(f"Items saved to: {ITEM_META_PARQUET}")
    print(f"Transactions saved to: {TRANSACTIONS_PARQUET}")
    print(f"Elapsed time:   {elapsed_time:.2f} seconds")
    print("=" * 70 + "\n")

    return stats


if __name__ == "__main__":
    run_preprocessing()
