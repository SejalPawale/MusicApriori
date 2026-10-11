"""
mining.py: Association Rule Mining Engine using Apriori and FP-Growth.

Handles:
- Loading basket representation (Sparse DataFrame or Sparse CSR matrix)
- Mining frequent itemsets with mlxtend.apriori or mlxtend.fpgrowth
- Generating association rules with confidence, lift, leverage, and conviction
- Version-safe handling of mlxtend's association_rules (num_itemsets parameter)
- Runtime measurement and algorithm benchmarking
"""

import sys
import time
import inspect
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import load_npz

import json

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BASKET_PARQUET, BASKET_SPARSE_NPZ, BASKET_COLUMNS_JSON, ITEM_META_PARQUET, TRANSACTIONS_PARQUET


def load_basket_data():
    """
    Loads transaction basket data into memory.
    Prefers basket_columns.json with basket_sparse.npz to guarantee column order.
    Returns: (basket_df, num_transactions, item_names)
    """
    if BASKET_SPARSE_NPZ.exists() and BASKET_COLUMNS_JSON.exists():
        X = load_npz(BASKET_SPARSE_NPZ)
        with open(BASKET_COLUMNS_JSON, "r", encoding="utf-8") as f:
            item_names = json.load(f)
        basket_df = pd.DataFrame.sparse.from_spmatrix(X, columns=item_names)
        return basket_df, X.shape[0], item_names

    if BASKET_PARQUET.exists():
        basket_df = pd.read_parquet(BASKET_PARQUET)
        return basket_df, len(basket_df), list(basket_df.columns)

    raise FileNotFoundError("Processed basket data not found! Please run 02_preprocess.py first.")


def mine_frequent_itemsets(basket_df, min_support=0.002, max_len=3, algorithm="fpgrowth", low_memory=True):
    """
    Mines frequent itemsets using either 'apriori' or 'fpgrowth'.

    Returns:
        (frequent_itemsets_df, elapsed_time_seconds)
    """
    from mlxtend.frequent_patterns import apriori, fpgrowth

    algo_func = apriori if algorithm.lower() == "apriori" else fpgrowth

    t0 = time.perf_counter()

    # Pre-prune columns whose 1-item support < min_support:
    # By the Apriori property, items below min_support can never be in any frequent itemset.
    # Pruning them and passing a compact boolean DataFrame provides a massive speedup in mlxtend.
    N = len(basket_df)
    min_count = min_support * N

    try:
        if hasattr(basket_df, "sparse"):
            csr = basket_df.sparse.to_coo().tocsr()
            counts = csr.sum(axis=0).A1
            valid_idx = [i for i, cnt in enumerate(counts) if cnt >= min_count]
            if not valid_idx:
                return pd.DataFrame(), time.perf_counter() - t0
            sub_arr = csr[:, valid_idx].toarray()
            cols = list(basket_df.columns)
            valid_cols = [cols[i] for i in valid_idx]
            input_df = pd.DataFrame(sub_arr, columns=valid_cols)
        else:
            input_df = basket_df
    except Exception:
        input_df = basket_df

    if algorithm.lower() == "apriori":
        sig = inspect.signature(apriori)
        kwargs = {"min_support": min_support, "use_colnames": True, "max_len": max_len}
        if "low_memory" in sig.parameters:
            kwargs["low_memory"] = low_memory
        frequent_itemsets = algo_func(input_df, **kwargs)
    else:
        frequent_itemsets = algo_func(input_df, min_support=min_support, use_colnames=True, max_len=max_len)

    elapsed = time.perf_counter() - t0

    if not frequent_itemsets.empty:
        frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
        frequent_itemsets["playlist_count"] = (frequent_itemsets["support"] * len(basket_df)).round().astype(int)
        frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False).reset_index(drop=True)

    return frequent_itemsets, elapsed


def generate_rules(frequent_itemsets, num_transactions, min_confidence=0.25, min_lift=1.0):
    """
    Extracts association rules from frequent itemsets.
    Handles compatibility with both older and newer mlxtend versions.

    Returns:
        rules_df: DataFrame of filtered association rules
    """
    if frequent_itemsets is None or frequent_itemsets.empty:
        return pd.DataFrame()

    from mlxtend.frequent_patterns import association_rules

    sig = inspect.signature(association_rules)
    kwargs = {
        "df": frequent_itemsets,
        "metric": "confidence",
        "min_threshold": min_confidence
    }
    if "num_itemsets" in sig.parameters:
        kwargs["num_itemsets"] = num_transactions

    try:
        rules = association_rules(**kwargs)
    except TypeError:
        # Fallback if argument mismatch
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if rules.empty:
        return pd.DataFrame()

    # Filter by minimum lift
    rules = rules[rules["lift"] >= min_lift].copy()

    if rules.empty:
        return pd.DataFrame()

    # Calculate playlist co-occurrence count
    rules["co_occurrence_count"] = (rules["support"] * num_transactions).round().astype(int)

    # Convert frozensets to displayable strings
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: " + ".join(list(s)))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: " + ".join(list(s)))

    # Ensure leverage and conviction are present
    if "leverage" not in rules.columns:
        rules["leverage"] = rules["support"] - (rules["antecedent support"] * rules["consequent support"])
    if "conviction" not in rules.columns:
        # conviction = (1 - consequent support) / (1 - confidence)
        rules["conviction"] = rules.apply(
            lambda r: (1 - r["consequent support"]) / (1 - r["confidence"] + 1e-9) if r["confidence"] < 1.0 else np.nan,
            axis=1
        )

    rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)
    return rules


def run_mining_pipeline(min_support=0.002, min_confidence=0.25, min_lift=1.5, max_len=3, algorithm="fpgrowth"):
    """
    Executes mining pipeline end-to-end.
    """
    basket_df, num_transactions, items = load_basket_data()
    print(f"[mining] Mining on {num_transactions:,} transactions across {len(items):,} items...")
    print(f"[mining] Settings: algorithm={algorithm}, min_support={min_support:.4%} (min {int(np.ceil(min_support * num_transactions))} playlists), min_conf={min_confidence}, min_lift={min_lift}, max_len={max_len}")

    fi, fi_time = mine_frequent_itemsets(
        basket_df,
        min_support=min_support,
        max_len=max_len,
        algorithm=algorithm
    )
    print(f"[mining] Found {len(fi):,} frequent itemsets in {fi_time:.3f}s")

    t_rules_0 = time.perf_counter()
    rules = generate_rules(
        fi,
        num_transactions=num_transactions,
        min_confidence=min_confidence,
        min_lift=min_lift
    )
    rules_time = time.perf_counter() - t_rules_0
    print(f"[mining] Found {len(rules):,} association rules in {rules_time:.3f}s")

    return fi, rules, fi_time + rules_time
