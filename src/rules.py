"""
rules.py: Step 4 - Plain-language Rule Explanations and Ranking Filters.

Implements:
1. Plain-language sentence generation for each association rule:
   "Playlists containing A also contain B in 38% of cases (confidence). That is 24× more often than expected from B’s overall popularity (lift). The pair appears together in 212 playlists (support 0.21%)."
2. Same-artist rule filtering (hides rules where antecedent and consequent are the same artist)
3. Minimum co-occurrence count filtering (protects against inflated lift from tiny counts)
4. Popularity penalty (filters out ultra-popular consequents to uncover niche cross-artist associations)
5. Metric explanations: support, confidence, lift, leverage, conviction
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def generate_rule_explanation(row) -> str:
    """
    Generates a human-readable, plain-language explanation of an association rule.
    """
    antecedents = list(row["antecedents"]) if hasattr(row["antecedents"], "__iter__") else [str(row["antecedents"])]
    consequents = list(row["consequents"]) if hasattr(row["consequents"], "__iter__") else [str(row["consequents"])]

    a_str = " + ".join([f"'{item}'" for item in antecedents])
    b_str = " + ".join([f"'{item}'" for item in consequents])

    conf_pct = row["confidence"] * 100.0
    lift_val = row["lift"]
    supp_pct = row["support"] * 100.0
    co_occur = int(row["co_occurrence_count"]) if "co_occurrence_count" in row else None

    count_phrase = f"in {co_occur:,} playlists " if co_occur is not None else ""

    sentence = (
        f"Playlists containing {a_str} also contain {b_str} in {conf_pct:.1f}% of cases (confidence). "
        f"That is {lift_val:.1f}× more often than expected from {b_str}'s overall popularity (lift). "
        f"The pair appears together {count_phrase}(support {supp_pct:.2f}%)."
    )
    return sentence


def get_item_artist(item_str: str) -> str:
    """
    Extracts the artist component from an 'artist – track' string.
    """
    if " – " in item_str:
        return item_str.split(" – ")[0].strip().lower()
    return item_str.strip().lower()


def is_same_artist_rule(row) -> bool:
    """
    Checks if all items in antecedents and consequents share the exact same artist.
    """
    antecedent_artists = {get_item_artist(item) for item in row["antecedents"]}
    consequent_artists = {get_item_artist(item) for item in row["consequents"]}
    all_artists = antecedent_artists.union(consequent_artists)
    return len(all_artists) == 1


def filter_rules(
    rules_df: pd.DataFrame,
    hide_same_artist: bool = False,
    min_co_occurrence: int = 5,
    popularity_penalty_top_n: int = 0,
    top_popular_items: set = None
) -> pd.DataFrame:
    """
    Applies user-selected filters to surface high-quality, non-obvious association rules.

    Parameters:
        rules_df: DataFrame of association rules
        hide_same_artist: If True, filters out intra-artist rules (e.g. tracks from the same album)
        min_co_occurrence: Minimum number of playlists the itemset must co-occur in
        popularity_penalty_top_n: Number of top global items to exclude from consequents (0 = disabled)
        top_popular_items: Set of item strings representing the top N popular items
    """
    if rules_df.empty:
        return rules_df

    filtered = rules_df.copy()

    # 1. Minimum co-occurrence count
    if min_co_occurrence > 1 and "co_occurrence_count" in filtered.columns:
        filtered = filtered[filtered["co_occurrence_count"] >= min_co_occurrence]

    # 2. Hide same-artist rules
    if hide_same_artist and not filtered.empty:
        is_same = filtered.apply(is_same_artist_rule, axis=1)
        filtered = filtered[~is_same]

    # 3. Popularity penalty on consequents
    if popularity_penalty_top_n > 0 and top_popular_items and not filtered.empty:
        def has_top_consequent(row):
            consequents = set(row["consequents"])
            return any(item in top_popular_items for item in consequents)

        filtered = filtered[~filtered.apply(has_top_consequent, axis=1)]

    # 4. Generate plain language explanations
    if not filtered.empty:
        filtered["explanation"] = filtered.apply(generate_rule_explanation, axis=1)

    return filtered.reset_index(drop=True)
