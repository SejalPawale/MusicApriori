# DATA CARD: Larxel Spotify Playlists Dataset (TuneTrail)

## 1. Dataset Overview & Provenance
* **Dataset Name**: Spotify Playlists Dataset (Larxel / Kaggle)
* **Underlying Source**: Collected from `#nowplaying` tweets and published via Spotify Web API.
* **Academic Citation**: Pichl, M., Zangerle, E., & Specht, G. (2015). *Towards a context-aware music recommendation approach: What is the difference between playlists and listening sessions?* In IEEE International Conference on Data Mining Workshop (ICDMW 2015), pp. 1324–1331.
* **Licence**: Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Format**: Comma-Separated Values (CSV), quoted fields with backslash escape characters.

---

## 2. Core Concepts & Mapping

| Concept | Formal Definition | Rationale & Critical Notes |
| :--- | :--- | :--- |
| **Transaction** | One playlist, uniquely identified by composite key `(user_id, playlistname)` $\rightarrow$ `tid = user_id + "\|" + playlistname` | Using `playlistname` alone conflates generic names like *"Starred"*, *"Favorites"*, or *"My Playlist"* across distinct users into massive synthetic transactions. Always incorporate `user_id`. |
| **Item (Track-Level)** | One track, represented as a normalized string: `artist – track` | Merges slight metadata variants into a uniform token. |
| **Item (Artist-Level)** | One artist, represented as the normalized `artist` | Allows macro-level genre and artist co-occurrence discovery. |
| **Interpretation** | *"These items were placed in the same user-curated playlist"* | Reflects human curation and playlist context, **not** acoustical similarity. |

---

## 3. Schema

| Field Name | Type | Description | Cleaning Applied |
| :--- | :--- | :--- | :--- |
| `user_id` | String | Anonymized alphanumeric user hash | Missing values dropped; hashed identifiers respected. |
| `artistname` | String | Track artist name | Lowercased, stripped of leading/trailing whitespace, collapsed internal whitespace. |
| `trackname` | String | Song title | Lowercased, stripped of edition suffixes (`- Remastered`, `- Live`, `- Radio Edit`, etc.). |
| `playlistname` | String | Name assigned by user | Preserved for composite key; whitespace trimmed. |

---

## 4. Preprocessing & Filtering Pipeline

Every stage of the cleaning pipeline and its motivation:

1. **Missing Value Removal**:
   * Rows missing `user_id`, `artistname`, `trackname`, or `playlistname` are discarded immediately.
2. **Text Normalization & Suffix Stripping**:
   * Suffixes matching regex `\s*[-–/]\s*(remaster(ed)?(\s*\d{4})?|live|radio\s*edit|single\s*version|deluxe|bonus\s*track|acoustic|mono|stereo|original\s*mix|extended\s*mix)\b.*$` are pruned.
   * Merges fragmented variants (e.g. *"Wonderwall"* and *"Wonderwall - Remastered"*) into a single frequent token.
3. **Intra-Playlist Track De-duplication**:
   * Repeated appearances of the same track within a single playlist are collapsed to a single binary indicator $(tid, item)$.
4. **Playlist Length Boundaries**:
   * **Minimum length**: 2 tracks. Playlists with $< 2$ tracks cannot generate association rules ($k \ge 2$).
   * **Maximum length cap**: 99th percentile of playlist length distribution. Eliminates dump/archive playlists (1,000+ tracks) which cause combinatorial explosion in itemset candidate generation.
5. **Duplicate Playlist Removal per User**:
   * Users frequently clone or duplicate identical tracklists under differing names (e.g. *"Starred"* and *"Chill"*).
   * Filter deduplicates on the `frozenset` of items per user.
6. **Item Universe Floor & Top-K Cap**:
   * **Support Floor**: Minimum frequency threshold of $0.05\%$ ($0.0005 \times N$) of total playlists. Tracks occurring less than this floor mathematically cannot appear in frequent itemsets.
   * **Max Items**: Capped at the top 20,000 most frequent items to preserve sparse matrix efficiency and ensure fast runtime on commodity hardware ($8\text{–}16\text{ GB RAM}$).
7. **Re-filtering**:
   * Playlists whose surviving tracks drop below 2 after item universe filtering are removed.
8. **Sparse Binary Representation**:
   * Converted to Compressed Sparse Row (CSR) matrix and saved to Parquet and `.npz`.

---

## 5. Audio-Feature Clarification
The dataset contains **no native audio features** and track IDs are plain text names, not 128-bit Spotify URIs. Any audio feature enrichment (e.g., tempo, danceability, energy) joined via approximate name matching is strictly decorative and describes track characteristics—**never why tracks co-occur**.

---

## 7. Empirical Dataset Statistics (Real Run Results)

| Metric | Raw Dataset Count | After Preprocessing ($20\%$ User Sample) | Full Scale Projection ($100\%$) |
| :--- | :--- | :--- | :--- |
| **Total Rows Read** | 12,891,570 | 2,492,145 | 12,891,570 |
| **Missing Values Dropped** | 34,856 | 34,856 | 34,856 |
| **Unique Users** | 15,914 | 3,182 | 15,914 |
| **Initial Playlists (`tid`s)** | 231,575 | 47,043 | 231,575 |
| **99th Percentile Length Cap** | 627 tracks | 605 tracks | 627 tracks |
| **Playlists Removed by Length Cap** | - | 2,584 | ~12,900 |
| **Duplicate User Playlists Removed** | - | 85 | ~425 |
| **Unique Raw Tracks** | 2,789,529 | 949,626 | 2,789,529 |
| **Variant Tracks Merged (Regex)** | - | 49,836 | ~150,000 |
| **Final Mining Playlists ($N$)** | - | **23,205** | ~115,000 |
| **Final Mining Items ($M$)** | - | **8,364** | ~20,000 (capped) |
| **Final Transaction Cells** | - | 412,549 | ~2,000,000 |
| **Matrix Sparsity** | - | **99.79%** | ~99.8% |

