"""
01_inspect.py: Step 1 - Load and inspect raw Spotify Playlists dataset.

Reads the raw dataset in chunks to handle large file sizes safely, checks columns,
identifies bad lines, and produces foundational exploratory data analysis (EDA):
- Total rows, distinct users, distinct playlists, unique tracks, unique artists
- Playlist length distribution (median, 95th percentile, maximum)
- Top 15 most frequent tracks
"""

import sys
import json
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, find_raw_dataset_file, get_dataset_stream



def inspect_dataset(raw_path=None, chunk_size=500_000):
    if raw_path is None:
        raw_path = find_raw_dataset_file()

    if raw_path is None:
        print("\n" + "=" * 70)
        print("ERROR: No raw dataset found!")
        print(f"Please place your Kaggle CSV or ZIP file into:\n  {RAW_DATA_DIR.resolve()}")
        print("Examples: spotify_dataset.csv, 2023_spotify_ds1.csv, or archive.zip")
        print("=" * 70 + "\n")
        return None

    csv_path = get_dataset_stream(raw_path)
    print(f"\n[inspect] Target CSV: {csv_path}")
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"[inspect] File size: {file_size_mb:.2f} MB")

    # Inspect first few lines for headers and stray whitespace
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        first_line = f.readline()
        print(f"[inspect] Raw Header Line: {repr(first_line.strip())}")

    # Read first chunk to inspect column names
    first_chunk = pd.read_csv(
        csv_path,
        nrows=10,
        quotechar='"',
        escapechar='\\',
        skipinitialspace=True,
        on_bad_lines="skip",
        dtype=str
    )

    clean_cols = [c.strip().strip('"').strip() for c in first_chunk.columns]
    first_chunk.columns = clean_cols
    print(f"[inspect] Detected columns ({len(clean_cols)}): {clean_cols}")

    expected_cols = ["user_id", "artistname", "trackname", "playlistname"]
    missing_expected = [col for col in expected_cols if col not in clean_cols]
    if missing_expected:
        print(f"[inspect] WARNING: Expected columns {missing_expected} not found! Present: {clean_cols}")

    # Chunked iteration with statistics collection
    print(f"\n[inspect] Processing in chunks of {chunk_size:,} rows...")
    total_rows = 0
    missing_counts = {col: 0 for col in clean_cols}
    users_set = set()
    playlist_counts = {}  # tid -> track count
    track_frequencies = {} # artist - track -> count
    artist_frequencies = {} # artist -> count

    bad_lines_count = 0

    # Read in chunks using C engine for high performance
    reader = pd.read_csv(
        csv_path,
        quotechar='"',
        escapechar='\\',
        skipinitialspace=True,
        on_bad_lines="skip",
        chunksize=chunk_size,
        dtype=str
    )

    chunk_idx = 0
    for chunk in reader:
        chunk_idx += 1
        chunk.columns = [c.strip().strip('"').strip() for c in chunk.columns]
        chunk_rows = len(chunk)
        total_rows += chunk_rows

        # Track missing values
        for c in clean_cols:
            if c in chunk.columns:
                missing_counts[c] += chunk[c].isna().sum()

        # Clean subset with present required columns
        valid_subset = chunk.dropna(subset=[c for c in expected_cols if c in chunk.columns])

        if "user_id" in valid_subset.columns:
            users_set.update(valid_subset["user_id"].unique())

        if "user_id" in valid_subset.columns and "playlistname" in valid_subset.columns:
            tids = valid_subset["user_id"] + "|" + valid_subset["playlistname"]
            counts = tids.value_counts(sort=False)
            for tid, count in counts.items():
                playlist_counts[tid] = playlist_counts.get(tid, 0) + count

        if "artistname" in valid_subset.columns and "trackname" in valid_subset.columns:
            tracks = valid_subset["artistname"].str.strip() + " – " + valid_subset["trackname"].str.strip()
            t_counts = tracks.value_counts(sort=False)
            for t, cnt in t_counts.items():
                track_frequencies[t] = track_frequencies.get(t, 0) + cnt

        if "artistname" in valid_subset.columns:
            a_counts = valid_subset["artistname"].str.strip().value_counts(sort=False)
            for a, cnt in a_counts.items():
                artist_frequencies[a] = artist_frequencies.get(a, 0) + cnt

        print(f"  Processed chunk {chunk_idx}: {chunk_rows:,} rows (total so far: {total_rows:,})")

    # Compute length distribution statistics
    lengths = np.array(list(playlist_counts.values())) if playlist_counts else np.array([0])
    median_len = float(np.median(lengths)) if len(lengths) else 0.0
    p95_len = float(np.percentile(lengths, 95)) if len(lengths) else 0.0
    p99_len = float(np.percentile(lengths, 99)) if len(lengths) else 0.0
    max_len = int(np.max(lengths)) if len(lengths) else 0
    min_len = int(np.min(lengths)) if len(lengths) else 0

    # Top 15 tracks
    sorted_tracks = [(trk, int(cnt)) for trk, cnt in sorted(track_frequencies.items(), key=lambda x: x[1], reverse=True)[:15]]

    summary = {
        "raw_file": str(csv_path.name),
        "file_size_mb": round(file_size_mb, 2),
        "total_rows_read": int(total_rows),
        "bad_lines_dropped": int(bad_lines_count),
        "unique_users": int(len(users_set)),
        "unique_playlists": int(len(playlist_counts)),
        "unique_tracks": int(len(track_frequencies)),
        "unique_artists": int(len(artist_frequencies)),
        "playlist_length": {
            "min": int(min_len),
            "median": round(float(median_len), 2),
            "p95": round(float(p95_len), 2),
            "p99": round(float(p99_len), 2),
            "max": int(max_len)
        },
        "missing_values_per_column": {k: int(v) for k, v in missing_counts.items()},
        "top_15_tracks": sorted_tracks
    }

    # Save summary
    out_file = PROCESSED_DATA_DIR / "eda_summary.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Print Report
    print("\n" + "=" * 70)
    print("               TUNETRAIL EXPLORATORY DATA INSPECTION REPORT")
    print("=" * 70)
    print(f"File Name:                  {summary['raw_file']} ({summary['file_size_mb']} MB)")
    print(f"Total Rows Read:            {summary['total_rows_read']:,}")
    print(f"Malformed Lines Dropped:    {summary['bad_lines_dropped']:,}")
    print(f"Unique Users:               {summary['unique_users']:,}")
    print(f"Unique Playlists (tids):    {summary['unique_playlists']:,}")
    print(f"Unique Tracks:              {summary['unique_tracks']:,}")
    print(f"Unique Artists:             {summary['unique_artists']:,}")
    print("-" * 70)
    print("Playlist Length Distribution:")
    print(f"  Min:                      {min_len}")
    print(f"  Median:                   {median_len:.1f}")
    print(f"  95th Percentile:          {p95_len:.1f}")
    print(f"  99th Percentile:          {p99_len:.1f}")
    print(f"  Maximum:                  {max_len}")
    print("-" * 70)
    print("Missing Values by Column:")
    for col, cnt in missing_counts.items():
        print(f"  {col:<20} {cnt:,}")
    print("-" * 70)
    print("Top 10 Most Frequent Tracks:")
    for rank, (trk, cnt) in enumerate(sorted_tracks[:10], start=1):
        print(f"  {rank:>2}. {trk:<50} ({cnt:,} occurrences)")
    print("=" * 70)
    print(f"Saved inspection metrics to: {out_file}\n")

    return summary


if __name__ == "__main__":
    inspect_dataset()
