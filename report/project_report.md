# TuneTrail: Spotify Playlist Association Rule Mining
## Project Report & Technical Documentation

**Dataset:** Larxel Spotify Playlists Dataset (Kaggle / Pichl et al.)  
**Domain:** Unsupervised Machine Learning / Data Mining  
**Methods:** FP-Growth, Apriori, Association Rule Mining, Holdout & Permutation Null Validation  

---

## 1. Project Overview & Problem Formulation

### 1.1 Objective
Recommender systems frequently rely on collaborative filtering matrix factorization or acoustic audio similarity. However, audio features alone do not capture human listening context. When a listener creates a playlist, they curate songs around distinct moods, road trips, study sessions, workouts, or personal eras.

**TuneTrail** applies unsupervised Association Rule Mining (**FP-Growth** and **Apriori**) to discover co-occurrence relationships and affinity rules directly from user-curated Spotify playlists.

### 1.2 The Core Data Modeling Principle
The dataset contains four primary text fields: `user_id`, `artistname`, `trackname`, and `playlistname`.

A critical conceptual distinction governs this project:
> **"These items were placed in the same user-curated playlist"**, not *"these songs sound similar"*.

1. **Transaction Definition:** A transaction represents one playlist created by one specific user:
   $$\text{tid} = \text{user\_id} \mathbin{\Vert} \text{playlistname}$$
   *Why this matters:* If `playlistname` were used on its own, ubiquitous names such as "Starred", "Favorites", "Chill", or "Workout" would merge thousands of unrelated listeners into massive artificial "mega-transactions", creating false co-occurrences. Keying transactions by `(user_id, playlistname)` guarantees strict transaction integrity.
2. **Item Definition:** An item represents a canonical track token formatted as:
   $$\text{item} = \text{normalized\_artist} \mathbin{\Vert} \text{" – "} \mathbin{\Vert} \text{normalized\_track}$$
3. **No Track ID / No Native Audio Features:** The raw dataset contains textual track strings, not Spotify 22-character URIs. Transactions are formed entirely on empirical playlist co-presence.

---

## 2. Dataset Pipeline & Preprocessing Audit

### 2.1 Raw Dataset Audit
* **Total Rows Processed:** 12,891,570 track entries (1.13 GB CSV).
* **Unique Users:** 15,914.
* **Unique Playlists:** 231,575.
* **Bad / Corrupted Lines Dropped:** 0 (clean quote handling with backslash escape).
* **Missing Metadata Dropped:** 34,856 rows with missing artist or track names.

### 2.2 Cleaning & Transformation Steps
1. **Text Normalization & Noise Suffix Removal:** Track titles frequently contain edition noise that artificially splits identical tracks into distinct tokens (e.g., *"Hotel California"*, *"Hotel California - Remastered 2013"*, *"Hotel California - Live"*). We applied a regex cleaner:
   $$\text{Regex: } \verb|\s*[-–/]\s*(remaster(ed)?(\s*\d{4})?|live|radio\s*edit|single\s*version|deluxe|bonus|acoustic|mono|stereo)\b.*|$$
   This merged **49,836 redundant track variants** back into unified songs.
2. **Intra-Playlist Deduplication:** 29,626 duplicate song entries within the same playlist were collapsed to a single presence.
3. **Playlist Length Bounds:**
   * Bottom threshold: Removed playlists with $< 2$ items (a single song cannot form co-occurring itemsets).
   * Upper threshold: Capped at the 99th percentile ($605$ songs). This eliminated bulk discography dumps and library backup dumps that would otherwise generate an exponential number of spurious candidate pairs.
4. **Duplicate Playlist Removal:** Users often duplicate their own playlists under different names. 85 identical itemset playlists from the same user were pruned.
5. **Item Universe & Sparse Matrix:** Filtered items appearing in fewer than $0.05\%$ of playlists ($0.0005 \times N$). 
   * **Final Matrix:** $23,205$ playlists $\times$ $8,364$ frequent tracks in a SciPy Compressed Sparse Row (CSR) matrix.
   * **Matrix Sparsity:** $99.79\%$.

---

## 3. Mathematical Foundations of Association Rules

For any association rule $X \Rightarrow Y$ (where $X$ is the antecedent and $Y$ is the consequent):

1. **Support:** The proportion of total playlists containing both $X$ and $Y$:
   $$\text{Support}(X \cup Y) = \frac{|\{P \mid X \cup Y \subseteq P\}|}{N}$$
2. **Confidence:** The conditional probability that a playlist contains $Y$ given that it contains $X$:
   $$\text{Confidence}(X \Rightarrow Y) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)} = P(Y \mid X)$$
3. **Lift:** The ratio of observed joint occurrence to expected co-occurrence under statistical independence:
   $$\text{Lift}(X \Rightarrow Y) = \frac{\text{Confidence}(X \Rightarrow Y)}{\text{Support}(Y)} = \frac{\text{Support}(X \cup Y)}{\text{Support}(X) \cdot \text{Support}(Y)}$$
   * $\text{Lift} = 1$: $X$ and $Y$ appear together purely by random chance.
   * $\text{Lift} > 1$: Positive association (co-occur significantly more than expected).
   * $\text{Lift} < 1$: Negative association / substitution.
4. **Leverage:** Absolute difference between observed frequency and expected frequency under independence:
   $$\text{Leverage}(X \Rightarrow Y) = \text{Support}(X \cup Y) - \text{Support}(X) \cdot \text{Support}(Y)$$

### Hand-Computed Toy Dataset Verification
To verify mathematical correctness, the pipeline was benchmarked against a known 5-playlist, 6-item toy ground truth:
* $P_1 = \{A, B, C\}$, $P_2 = \{A, B\}$, $P_3 = \{A, C\}$, $P_4 = \{B, D\}$, $P_5 = \{E, F\}$ ($N = 5$)
* Rule $\{A\} \Rightarrow \{B\}$:
  * $\text{Support} = 2/5 = 0.4000$
  * $\text{Confidence} = (2/5) / (3/5) = 2/3 \approx 0.6667$
  * $\text{Lift} = (2/3) / (3/5) = 10/9 \approx 1.1111$
  * $\text{Leverage} = 0.40 - (0.60 \times 0.60) = +0.0400$
The test suite (`tunetrail/tests/test_correctness.py`) verified that the implementation matches these hand-computed values to 4 decimal places.

---

## 4. Workload Division (4-Person Team Allocation)

| Role | Person | Core Responsibilities | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Data Engineering & Preprocessing** | **Person 1** | • Chunked CSV ingestion & header whitespace auditing<br>• Text normalization regex & noise suffix stripping<br>• Composite transaction key generation (`tid = user \| playlist`)<br>• CSR sparse matrix serialization (`.npz`, `.parquet`) | • `01_inspect.py`<br>• `02_preprocess.py`<br>• `eda_summary.json`<br>• `DATA_CARD.md` |
| **Mining Algorithms & Optimization** | **Person 2** | • Implementing FP-Growth and Apriori with mlxtend<br>• Vectorized sparse column frequency pre-filtering<br>• Execution parameter sweep grid (`experiments.py`)<br>• Benchmarking runtime scaling vs. min_support thresholds | • `mining.py`<br>• `experiments.py`<br>• `sweep_results.csv`<br>• `sweep_runtime.png` |
| **Rule Filtering, Ranking & Evaluation** | **Person 3** | • Plain-language translation generator<br>• Same-artist suppression filter & co-occurrence floor<br>• Independent recount validation on raw transaction sets<br>• 80/20 train/test holdout validation & shuffled null baseline | • `rules.py`<br>• `evaluate.py`<br>• `03_mine_test.py`<br>• `test_correctness.py` |
| **Visualizations & Technical Reporting** | **Person 4** | • Generating distribution charts & top rules bar graphs<br>• Interactive Support vs. Confidence Plotly scatter plot<br>• NetworkX graph export (node size = count, edge = lift)<br>• Final report compilation, tabular summaries & review | • `viz.py`<br>• `viz_1` to `viz_4` HTML/PNG<br>• `sweep_heatmap.png`<br>• Project Report |

---

## 5. Detailed Explanation of the Visualizations

### Chart 1: Frequent Itemset Distribution by Size (`viz_1_itemsets_by_size.html`)
* **What it plots:** A vertical bar chart showing the count of discovered frequent itemsets categorized by itemset size $k$ ($k=1$ for single tracks, $k=2$ for pairs, $k=3$ for triplets).
* **Key Observations:**
  * At a support threshold of $0.5\%$, the algorithm discovered **472 singletons ($k=1$)** and **15 frequent pairs ($k=2$)**.
  * When support is lowered to $0.3\%$, singletons expand to 729 while pairs jump to **1,290** and triplets ($k=3$) emerge.
* **Significance:** Illustrates the Apriori downward closure property: an itemset of size $k$ can only be frequent if all its subsets are frequent. As itemset size increases, joint support drops sharply in sparse data.

### Chart 2: Top Association Rules Ranked by Lift (`viz_2_top_rules_bar.html`)
* **What it plots:** A horizontal bar chart ranking the top 15 discovered rules by their **Lift** metric, alongside confidence values.
* **Key Observations:**
  * Top rules show Lift values ranging between **$45.0\times$ and $63.0\times$**.
  * Prominent pairs include:
    * *Mumford & Sons – The Cave* $\Rightarrow$ *Mumford & Sons – Little Lion Man* ($\text{Lift} = 62.80$, $\text{Conf} = 60.89\%$)
    * *Arctic Monkeys – Do I Wanna Know?* $\Rightarrow$ *Arctic Monkeys – Why'd You Only Call Me When You're High?* ($\text{Lift} = 55.41$, $\text{Conf} = 40.59\%$)
    * *Nirvana – Smells Like Teen Spirit* $\Rightarrow$ *Nirvana – Come As You Are* ($\text{Lift} = 46.65$, $\text{Conf} = 39.20\%$)
  * Applying the cross-artist filter reveals strong cultural connections such as *Jessie J – Bang Bang* $\Rightarrow$ *Ariana Grande – Break Free* ($\text{Lift} = 63.16$, $\text{Conf} = 46.54\%$) and *Ariana Grande – Problem* $\Rightarrow$ *Iggy Azalea – Fancy* ($\text{Lift} = 35.82$, $\text{Conf} = 35.35\%$).
* **Significance:** Demonstrates the value of the Lift metric over raw confidence. Ubiquitous songs might have moderate confidence simply because they are everywhere; Lift filters out this popularity bias and surfaces genuine affinity.

### Chart 3: Support vs. Confidence Scatter Plot Colored by Lift (`viz_3_support_vs_confidence.html`)
* **What it plots:** A scatter plot where:
  * **X-axis:** Rule Support percentage ($0.30\%$ to $0.80\%$).
  * **Y-axis:** Rule Confidence percentage ($20\%$ to $80\%$).
  * **Point Color / Hover:** Coded by **Lift** value.
* **Key Observations:**
  * **High Lift, Low Support Region (Top-Left):** Niche song pairs and tight sub-genre clusters cluster here. They appear in a modest percentage of total playlists ($0.3\%\text{–}0.5\%$), but when the first song is present, the second song co-occurs with very high probability ($> 60\%$), driving Lift above $40\times$.
  * **Moderate Lift, Higher Support Region (Bottom-Right):** Broad mainstream pop tracks (e.g. *Avicii*, *Imagine Dragons*). They appear in more playlists overall ($> 0.6\%$), but their co-occurrence probability is more diffuse, resulting in lower Lift ($10\times\text{–}25\times$).
* **Significance:** Visualizes the tradeoff in association rule mining: high confidence does not guarantee high lift, and the most distinctive playlist affinities reside at lower support thresholds.

### Bonus / Graph Visualization: Co-Occurrence Rule Network (`viz_4_rules_network.png`)
* **Nodes:** Unique tracks (node size proportional to total playlist frequency).
* **Edges:** Association rules connecting songs (edge thickness and intensity proportional to Lift).
* **Musical Clusters:** Forms distinct, coherent community subgraphs: 2013 Indie Rock (*Arctic Monkeys*, *The Black Keys*), 90s Grunge (*Nirvana*, *Pearl Jam*), and 2014 Mainstream Pop (*Ariana Grande*, *Iggy Azalea*, *Jessie J*).

---

## 6. Model Evaluation & Statistical Verification

To guarantee that discovered rules represent genuine patterns and not random data artifacts, four verification protocols were executed:

### 6.1 Independent Recount Verification
* **Method:** Sampled 20 rules generated by `mlxtend` and independently recomputed Support, Confidence, and Lift directly from raw playlist sets using pure set intersections ($\cap$) without machine learning libraries.
* **Result:** **0 mismatches across all 20 rules (100% exact numerical match)**.

### 6.2 80/20 Holdout Validation
* **Method:** Split the $23,205$ playlists into an $80\%$ training set ($18,564$ playlists) and a $20\%$ unseen test set ($4,641$ playlists). Mined rules on the training set and measured whether they maintained $\text{Lift} > 1.0$ on the test set.
* **Result:** **100.0% of discovered rules retained $\text{Lift} > 1.0$ on unseen playlists**, with an average confidence drop of only $0.0245$ ($2.45\%$), confirming robust generalization.

### 6.3 Shuffled Null Baseline (Permutation Test)
* **Method:** Randomly permuted songs across playlists while keeping playlist lengths exactly identical, then ran the mining engine at the exact same thresholds.
* **Result:** **0 frequent itemsets and 0 rules discovered**. In randomized data, songs do not co-occur above minimum support, proving that the real dataset reflects genuine human curation.

---

## 7. Key Findings, Limitations & Conclusion

### 7.1 Key Empirical Takeaways
1. **FP-Growth vs. Apriori:** FP-Growth executed in **~1.6 seconds** compared to Apriori's slower candidate generation passes on larger item universes.
2. **Support Threshold Tradeoff:** At $\text{min\_support} \ge 0.5\%$, only 24 rules were discovered (mostly intra-artist discography pairs). Lowering the floor to $0.3\%$ expanded the rule set to **2,486 rules** and uncovered **154 cross-artist relationships**.
3. **Lift Prevents Popularity Traps:** A song like *Imagine Dragons – Radioactive* appears in over $2,300$ playlists, giving it moderate co-occurrence with almost everything. Filtering by Lift $> 2.0$ effectively suppresses this background noise.

### 7.2 Practical Limitations
1. **Self-Selection Sample:** Playlists originate from public `#nowplaying` tweets on Spotify, skewing toward active social media users and popular English-language releases from 2011–2015.
2. **Intra-Artist Bias:** Users frequently save entire albums in one click. Enabling the **"Hide Same-Artist"** filter is necessary to surface cross-artist discoveries.
3. **Non-Causality:** Association rules show statistical co-presence, not causal intent.

### 7.3 Conclusion
TuneTrail demonstrates that unsupervised association rule mining can extract contextual, genre-consistent music affinities from raw playlist data. By enforcing composite transaction keys, normalizing track strings, benchmarking against null baselines, and ranking by Lift, the pipeline produces high-confidence, interpretable recommendations.
