# TuneTrail: Frequent Itemset Mining and Association Rule Analysis

TuneTrail is a machine learning project that discovers songs that frequently appear together in user-curated Spotify playlists. It treats each playlist as a transaction and applies **Apriori** as the primary frequent-itemset mining algorithm. The resulting itemsets are used to generate association rules that can be evaluated using support, confidence, lift and leverage.

**FP-Growth** is included as an alternative algorithm for comparison. The project focuses on interpreting playlist co-occurrence patterns rather than claiming that songs are acoustically similar or that one song causes a listener to prefer another.

## Project Structure

```text
tunetrail/
├── README.md                     # Project documentation
├── requirements.txt              # Project dependencies
├── .gitignore                    # Git rules for ignored files
├── data/
│   ├── raw/                      # Raw Kaggle CSV or ZIP dataset
│   ├── processed/                # Processed data and transaction matrices
│   └── DATA_CARD.md              # Dataset source, schema, and filters
├── src/
│   ├── config.py                 # Configuration parameters and paths
│   ├── 01_inspect.py             # Raw data inspection and exploration
│   ├── 02_preprocess.py          # Data cleaning and matrix generation
│   ├── preprocess.py             # Preprocessing pipeline
│   ├── 03_mine_test.py           # Quick mining and recount verification
│   ├── mining.py                 # Apriori and FP-Growth algorithms
│   ├── rules.py                  # Association rule generation and ranking
│   ├── viz.py                    # Visualizations and rule networks
│   ├── experiments.py            # Parameter sweeps and runtime benchmarks
│   ├── evaluate.py               # Correctness and robustness evaluation
│   └── generate_synthetic_data.py # Synthetic sample data generator
├── tests/
│   └── test_correctness.py       # Tests using hand-calculated examples
├── results/                      # Experiment outputs and visualizations

```

## Table of Contents

- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Methodology](#methodology)
  - [Apriori](#apriori)
  - [Association Rule Measures](#association-rule-measures)
  - [FP-Growth Comparison](#fp-growth-comparison)
- [Evaluation](#evaluation)
- [Interpreting Results](#interpreting-results)
- [Limitations](#limitations)
- [Technologies](#technologies)
- [References](#references)

## Project Objectives

- Convert playlist records into transactions suitable for association-rule mining.
- Discover frequent tracks, track pairs and larger combinations using Apriori.
- Generate readable association rules from frequent itemsets.
- Evaluate patterns using support, confidence, lift, and leverage.
- Explore how mining thresholds affect the number and strength of discovered patterns.
- Compare Apriori with FP-Growth under equivalent settings and check the correctness of mined results.

## Dataset

The project is designed to use the **Larxel Spotify Playlists dataset** available through Kaggle. Playlist records provide the information needed to identify playlists and their artist–track entries.

A playlist is identified using the composite key `(user_id, playlistname)`. This avoids merging playlists belonging to different users merely because they share a generic name such as `Favorites`.

> The exact number of transactions, itemsets, rules, and execution times depends on the dataset version, the amount of data processed, preprocessing choices, and the selected thresholds. Results should be reported alongside these settings.

## Data Preparation

Before mining, playlist records are cleaned and converted into transaction data.

1. **Identify playlists:** Use `(user_id, playlistname)` as the transaction identifier.
2. **Represent tracks as items:** Each item is a normalized artist–track pair.
3. **Normalize text:** Standardize track labels and, where appropriate, remove edition suffixes such as “Remastered” or “Live”.
4. **Handle duplicates and outliers:** Remove duplicate playlist entries and apply any selected filtering of unusually large playlists.
5. **Build the transaction representation:** Convert playlist membership into a sparse matrix, where each row represents a playlist and each column represents a track item.

Normalization can merge distinct recordings or editions into one item. This can improve consistency but may also hide differences between versions, so results should be interpreted with that trade-off in mind.

## Methodology

### Apriori

Apriori is the main algorithm used to discover frequent itemsets. It proceeds level by level:

1. Count individual tracks and retain those meeting the minimum-support threshold.
2. Generate candidate pairs from frequent single items and count their occurrences.
3. Retain frequent pairs and use them to form candidates of larger itemsets.
4. Prune candidates if any of their non-empty subsets are infrequent.
5. Continue until no larger frequent itemsets remain.

Apriori uses the principle that **every subset of a frequent itemset must also be frequent**. Therefore, a candidate with an infrequent subset can be discarded without counting it.

For an itemset \(X\), support is:

\[
\operatorname{support}(X)=
\frac{\text{number of playlists containing }X}
{\text{total number of playlists}}
\]

For example, a minimum support of `0.5%` means an itemset must occur in at least 5 out of every 1,000 playlist transactions to qualify. Lower support can reveal less common combinations but generally increases the number of candidates. Higher support reduces the search space but can exclude niche associations.

### Association Rule Measures

A rule has the form **A → B**, where `A` is the antecedent and `B` is the consequent. It describes a co-occurrence relationship in the playlist data.

| Measure | Formula | Interpretation |
|---|---|---|
| **Support** | `support(A ∪ B)` | Proportion of all playlists containing both A and B. |
| **Confidence** | `support(A ∪ B) / support(A)` | Proportion of playlists containing A that also contain B. |
| **Lift** | `confidence(A → B) / support(B)` | Compares the observed conditional frequency of B with B's overall frequency. |
| **Leverage** | `support(A ∪ B) − support(A) × support(B)` | Difference between observed co-occurrence and the co-occurrence expected under independence. |

General interpretation:

- **Lift greater than 1:** A and B occur together more often than expected from their individual frequencies.
- **Lift near 1:** The rule shows little association beyond the baseline frequency of B.
- **Lift below 1:** A and B co-occur less often than expected under independence.
- **High confidence alone is not enough:** A consequent that appears in many playlists may produce high confidence even when the association is not especially distinctive. Lift and support provide additional context.

#### Example from the playlist analysis

The documented analysis includes the rule:

**Oasis – Wonderwall → The Verve – Bitter Sweet Symphony**

The reported confidence is **38.0%**, meaning that 38% of playlists containing “Wonderwall” also contain “Bitter Sweet Symphony.” The reported lift is **24.0**, indicating a strong association relative to the consequent's overall frequency. The pair occurs in **212 playlists**, with reported support of approximately **0.21%**.

This example illustrates why measures should be considered together: the rule has high lift but relatively low overall support. It is an interesting association in a smaller share of playlists, not a combination that appears in most playlists.

### FP-Growth Comparison

FP-Growth is included as a secondary algorithm for comparison. Instead of generating candidate itemsets level by level, it compresses transaction data into an FP-tree and mines frequent patterns from that structure.

For a fair comparison, Apriori and FP-Growth should use the same prepared transactions and minimum-support threshold. Their frequent itemsets can be compared for agreement and execution time can be recorded as a practical efficiency measure. The relative performance depends on the data and settings; no algorithm should be assumed to be faster in every case.

## Evaluation

The project can evaluate patterns through threshold experiments and correctness checks.

### Threshold experiments

Example settings used for exploration include:

- **Minimum support:** `0.3%`, `0.5%`, `0.8%`, and `1.0%`
- **Minimum confidence:** `0.1`, `0.3`, `0.5`, and `0.7`
- **Minimum lift:** `1`, `2`, and `5`

Increasing minimum support usually reduces the number of itemsets retained. Increasing minimum confidence or lift filters the rules more strictly. The final number of patterns depends on the dataset and the combination of thresholds.

### Correctness and robustness checks

- **Hand-calculated examples:** Verify measure calculations on small, understandable transactions.
- **Independent recounting:** Check whether mined itemsets occur in the claimed number of playlists.
- **Holdout check:** Compare patterns across an 80/20 data split to examine whether they persist beyond the data used to discover them.
- **Shuffled baseline:** Disrupt the original co-occurrence structure and compare results with the unshuffled data.

These checks complement the association measures. A rule with high lift may be less useful if it is unstable or cannot be independently verified.

## Interpreting Results

Useful outputs for exploring the patterns include:

- Itemset counts grouped by itemset size.
- Ranked association rules, including rules with high lift.
- Support-versus-confidence plots, with lift used as an additional visual dimension.
- Rule-network visualizations showing connections among tracks.

These views help explore patterns but do not independently establish that an association is meaningful. Cross-artist rules can be especially useful when the goal is to discover connections beyond songs by the same artist.

## Limitations

- Playlist co-occurrence does not prove that one song causes a preference for another.
- Playlist context, curator choices, and track popularity can influence the rules.
- Normalizing titles can merge distinct recordings or editions.
- Sparse data and strict support thresholds can hide rare associations.
- Results depend on data sampling, preprocessing, and parameter choices.
- Runtime comparisons are only meaningful when the dataset, thresholds, and computing environment are held consistent.

## Technologies

The project is implemented in Python and uses the `mlxtend` frequent-pattern mining and association-rule functionality. Data preparation and analysis use tabular and sparse transaction representations, while visualizations support inspection of the mined patterns.

## References

1. Larxel, **Spotify Playlists dataset**, Kaggle.
2. **mlxtend documentation** — frequent-pattern mining and association rules: https://rasbt.github.io/mlxtend/
