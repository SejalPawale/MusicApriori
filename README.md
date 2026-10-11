# TuneTrail: Spotify Playlist Association Rule Mining 🎵

TuneTrail discovers musical affinities, co-occurrence patterns, and surprising cross-artist connections within user-curated Spotify playlists using Association Rule Mining (**Apriori** & **FP-Growth**).

---

## Project Structure

```
tunetrail/
├── README.md                      # Project documentation and quickstart
├── requirements.txt               # Pinned dependencies
├── .gitignore                     # Git rules (raw datasets & caches ignored)
├── data/
│   ├── raw/                       # Place Kaggle CSV or ZIP here
│   ├── processed/                 # Parquet outputs, CSR matrices, item metadata
│   └── DATA_CARD.md               # Dataset provenance, schema, and filters
├── src/
│   ├── config.py                  # Centralized tuning parameters & paths
│   ├── 01_inspect.py              # Step 1: Chunked inspection & raw EDA
│   ├── 02_preprocess.py           # Step 2: Text cleaning, filtering, matrix generation
│   ├── preprocess.py              # Full preprocessing pipeline module
│   ├── 03_mine_test.py            # Step 3: Quick CLI mining & recount verification
│   ├── mining.py                  # Apriori & FP-Growth mining engine
│   ├── rules.py                   # Plain-language generator & ranking filters
│   ├── viz.py                     # Visualizations generator (Plotly, NetworkX)
│   ├── experiments.py             # Parameter sweep grid & runtime benchmarks
│   ├── evaluate.py                # Correctness tests, holdout, shuffled baseline
│   └── generate_synthetic_data.py # Sample data generator for immediate testing
├── tests/
│   └── test_correctness.py        # Pytest suite with toy hand-computed assertions
├── results/                       # CSV sweeps, HTML charts, NetworkX graph figures
└── report/
    └── report.md                  # Comprehensive academic/project report
```

---

## Quickstart Guide

### 1. Environment Setup

Activate the virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r tunetrail/requirements.txt
```

### 2. Dataset Setup
Download the **Larxel Spotify Playlists** dataset from Kaggle and place either the `.zip` file or the extracted `.csv` directly into:
```
tunetrail/data/raw/
```
*(If you do not have the raw Kaggle dataset ready yet, run `python src/generate_synthetic_data.py` to test with realistic synthetic playlists).*

### 3. Step 1: Inspect Raw Data
Run chunked inspection to check header whitespace, bad lines, playlist length percentiles, and top tracks:
```bash
cd tunetrail/src
python 01_inspect.py
```

### 4. Step 2: Clean and Preprocess
Normalize track titles (stripping "- Remastered", "- Live"), filter out duplicate playlists, remove playlist dump outliers, and construct the sparse CSR matrix:
```bash
python 02_preprocess.py
```
*Note: In `src/config.py`, `SAMPLE_PCT = 20` by default for fast development. Set to `100` for full dataset execution.*

### 5. Step 3: Quick Mining & Recount Test
Run a quick frequent itemset and association rule extraction with automatic independent recount verification:
```bash
python 03_mine_test.py 0.005
```

### 6. Step 4: Run Evaluation & Unit Tests
Run the automated pytest test suite and evaluation suite (including the hand-computed toy dataset test, independent recounts, 80/20 holdout validation, and shuffled null baseline):
```bash
pytest ../tests
python evaluate.py
```

### 7. Step 5: Parameter Sweep Benchmarks
Execute grid sweep across Support $\in \{0.3\%, 0.5\%, 0.8\%, 1.0\%\}$, Confidence $\in \{0.1, 0.3, 0.5, 0.7\}$, and Lift $\in \{1, 2, 5\}$, benchmarking FP-Growth vs. Apriori runtimes:
```bash
python experiments.py
```

### 8. Step 6: Generate Visualizations
Generate all 4 project visualizations (Itemsets by size, Top rules by lift, Interactive Support vs. Confidence scatter, and NetworkX rule co-occurrence graph):
```bash
python viz.py
```
Output files are saved to `tunetrail/results/`:
- `viz_1_itemsets_by_size.html`
- `viz_2_top_rules_bar.html`
- `viz_3_support_vs_confidence.html` (interactive Plotly scatter colored by Lift)
- `viz_4_rules_network.html` & `viz_4_rules_network.png` (NetworkX graph)

---

## Core Concepts & Plain-Language Explanations

* **Transaction Key**: Composite key `user_id + "|" + playlistname` prevents false merging of distinct users' generic playlists (*"Starred"*, *"Favorites"*).
* **Track Token**: Cleaned `artist + " – " + track`.
* **Plain-Language Rule Explanation**:
  > *"Playlists containing 'oasis – wonderwall' also contain 'the verve – bitter sweet symphony' in 38.0% of cases (confidence). That is 24.0× more often than expected from the latter's overall popularity (lift). The pair appears together in 212 playlists (support 0.21%)."*
* **Surprising Patterns**: Toggle **"Hide same-artist rules"** to suppress intra-album rules (e.g. Queen $\rightarrow$ Queen) and uncover cross-artist musical affinities!
