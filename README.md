# TuneTrail: Frequent Itemset Mining and Association Rule Analysis 🎵

TuneTrail discovers musical affinities, co-occurrence patterns, and surprising cross-artist connections within user-curated Spotify playlists using Association Rule Mining (**Apriori** & **FP-Growth**). It treats each playlist as a transaction and applies frequent-itemset mining to generate association rules evaluated by Support, Confidence, Lift, and Leverage.

---

## ⚡ Quickstart: Clone & Run in 3 Minutes

If someone clones this repository (`git clone https://github.com/tanishaacodes/MusicApriori.git`), here is exactly how to set it up and run it immediately on any device (Mac, Windows, or Linux).

### 1. Clone & Set Up Virtual Environment

```bash
# 1. Clone the repository
git clone https://github.com/tanishaacodes/MusicApriori.git
cd MusicApriori

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate          # On Mac / Linux
# .venv\Scripts\activate           # On Windows Command Prompt / PowerShell

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📂 Dataset Guide (Why and How It Works)

The raw Kaggle Spotify dataset is **1.13 GB** (`spotify_dataset.csv`), which is too large to host inside a GitHub repository. 

### Option A: Immediate Zero-Setup Demo (No Download Needed!)
**You do not need to download the 1.13 GB file to run or test the project!**  
The repository already includes the **preprocessed sparse matrix** (`data/processed/basket_sparse.npz` and `data/processed/transactions.parquet`). You can run mining, unit tests, and visualization generation immediately:

```bash
# Verify unit tests & hand-calculated math (Runs in ~1 sec)
pytest tests/

# Run mining live on the dataset
python src/03_mine_test.py 0.005

# Run statistical evaluation (Recount, 80/20 Holdout, Shuffled Null Baseline)
python src/evaluate.py
```

---

### Option B: Running Step 1 & 2 with the Full 1.13 GB Kaggle Dataset
If you want to re-run the raw data cleaning and preprocessing pipeline from scratch:

1. Download the **Spotify Playlists Dataset** from Kaggle:  
   🔗 [Kaggle: Spotify Playlists (Larxel / Andrew MVD)](https://www.kaggle.com/datasets/andrewmvd/spotify-playlists)
2. Extract or place the file directly into `data/raw/`:
   ```
   data/raw/spotify_dataset.csv
   ```
3. Run Step 1 (Chunked EDA) & Step 2 (Preprocessing):
   ```bash
   python src/01_inspect.py
   python src/02_preprocess.py
   ```

*(Note: If you run `01_inspect.py` or `02_preprocess.py` without the Kaggle file present, the scripts will automatically inform you and offer to generate a synthetic test dataset for you!)*

---

## 🚀 Step-by-Step Execution Workflow

Run these commands in order:

### Step 1: Run Correctness & Validation Suite
```bash
pytest tests/
python src/evaluate.py
```
- **Toy Dataset Test**: Validates mathematical formulas against a hand-calculated 5-playlist, 6-item dataset to 4 decimal places.
- **Independent Recount**: Samples 20 random rules and recomputes support, confidence, and lift directly from raw playlist sets without ML libraries (**0 mismatches**).
- **80/20 Holdout Validation**: Mines on 80% of playlists and confirms that **100% of discovered rules maintain $\text{Lift} > 1.0$** on unseen playlists.
- **Shuffled Baseline**: Permutes track assignments; yields 0 rules, confirming patterns are genuine human curation.

### Step 2: Live Association Mining & Plain-Language Rules
```bash
python src/03_mine_test.py 0.005
```
Extracts frequent itemsets and prints plain-English explanations:
> *"Playlists containing 'Mumford & Sons – The Cave' also contain 'Mumford & Sons – Little Lion Man' in 60.9% of cases (confidence). That is 62.8× more often than expected from overall popularity (lift). The pair appears together in 123 playlists (support 0.53%)."*

### Step 3: View & Generate Visualizations
```bash
python src/viz.py
```
Exports 4 clean, academic visualizations to `results/`:
- `results/viz_1_itemsets_by_size.html`: Distribution of frequent singletons, pairs ($k=2$), and triplets ($k=3$).
- `results/viz_2_top_rules_bar.html`: Top 15 association rules ranked by Lift.
- `results/viz_3_support_vs_confidence.html`: Support vs. Confidence scatter plot with color-coded Lift.
- `results/viz_4_rules_network.png` & `.html`: Network co-occurrence graph showing music community clusters.

To open them on your computer:
```bash
open results/viz_2_top_rules_bar.html
open results/viz_3_support_vs_confidence.html
open results/viz_4_rules_network.png
```

### Step 4: Run Algorithmic Benchmark Sweeps
```bash
python src/experiments.py
```
Benchmarks **FP-Growth vs. Apriori** across Support, Confidence, and Lift thresholds. Saves `sweep_results.csv`, `sweep_heatmap.png`, and `sweep_runtime.png`.

---

## 📁 Repository Structure

```text
tunetrail/
├── README.md                      # Project documentation and quickstart guide
├── requirements.txt               # Pinned dependencies
├── .gitignore                     # Git rules (raw datasets & caches ignored)
├── data/
│   ├── raw/                       # Place Kaggle CSV or ZIP here (optional)
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
    ├── project_report.md          # Clean academic project report
    └── report.md                  # Detailed technical report
```

---

## 🧠 Core Methodology & Measures

### 1. Data Representation & Integrity
- **Composite Transaction Key**: Keying transactions by $\text{tid} = \text{user\_id} \mathbin{\Vert} \text{playlistname}$ ensures distinct users' playlists named *"Starred"* or *"Favorites"* are not merged into artificial mega-playlists.
- **Track Normalization**: Regex removes noisy track suffixes like `"- Remastered 2011"`, `"- Live"`, and `"- Radio Edit"`, consolidating 49,836 duplicate track variants into canonical artist–track tokens.

### 2. Association Rule Measures
For a rule **A → B**:
- **Support**: Proportion of all playlists containing both A and B: $\frac{|\{P \mid A \cup B \subseteq P\}|}{N}$.
- **Confidence**: Conditional probability of B given A: $\frac{\text{Support}(A \cup B)}{\text{Support}(A)}$.
- **Lift**: Ratio of observed co-occurrence to expected co-occurrence under independence: $\frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}$.
- **Leverage**: $\text{Support}(A \cup B) - \text{Support}(A) \times \text{Support}(B)$.

### 3. Intra-Artist vs. Cross-Artist Discovery
Users often save full albums (e.g., Queen discographies), causing top rules to be dominated by the same artist. Enabling the **"Hide same-artist"** filter in `rules.py` suppresses trivial pairs and uncovers cross-artist affinities (e.g. *Jessie J* $\rightarrow$ *Ariana Grande*).

---

## 📚 References
1. Pichl, M., Zangerle, E., & Specht, G. (2015). *Towards a context-aware music recommendation approach: What is the difference between playlists and listening sessions?* IEEE ICDMW.
2. Larxel / Andrew MVD, **Spotify Playlists dataset**, Kaggle (CC BY 4.0).
3. **mlxtend documentation**: https://rasbt.github.io/mlxtend/
