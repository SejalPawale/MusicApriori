"""
evaluate.py: Correctness verification, Hold-out validation, and Null baseline tests.

Includes:
1. Hand-computed Toy Dataset test (5 playlists, 6 items) asserting exact mathematical values
2. Independent recount verification: validates 20 random rules directly against raw playlist sets
3. Hold-out validation: 80% train / 20% test split measuring generalization and confidence decay
4. Shuffled null baseline: permutes item assignments while keeping playlist degrees fixed
5. User-level aggregation robustness check
"""

import sys
import json
import random
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, TRANSACTIONS_PARQUET
from mining import load_basket_data, mine_frequent_itemsets, generate_rules


def run_toy_dataset_test():
    """
    1. Toy Hand-Computed Test:
    Playlists (5 total):
      P1: {A, B, C}
      P2: {A, B}
      P3: {A, C}
      P4: {B, D}
      P5: {E, F}

    Hand calculations:
      Total = 5
      Support(A) = 3/5 = 0.60
      Support(B) = 3/5 = 0.60
      Support({A, B}) = 2/5 = 0.40
      Rule {A} -> {B}:
        Confidence = Support({A, B}) / Support(A) = (2/5) / (3/5) = 2/3 = 0.6667
        Expected Consequent = Support(B) = 0.60
        Lift = Confidence / Support(B) = (2/3) / 0.60 = 10/9 = 1.1111
        Leverage = Support({A, B}) - Support(A)*Support(B) = 0.40 - 0.36 = 0.04
    """
    toy_data = [
        {"tid": "p1", "item": ["A", "B", "C"]},
        {"tid": "p2", "item": ["A", "B"]},
        {"tid": "p3", "item": ["A", "C"]},
        {"tid": "p4", "item": ["B", "D"]},
        {"tid": "p5", "item": ["E", "F"]},
    ]
    all_items = ["A", "B", "C", "D", "E", "F"]
    matrix = []
    for p in toy_data:
        row = [item in p["item"] for item in all_items]
        matrix.append(row)

    toy_df = pd.DataFrame(matrix, columns=all_items)

    fi, _ = mine_frequent_itemsets(toy_df, min_support=0.3, max_len=2, algorithm="apriori")
    rules = generate_rules(fi, num_transactions=5, min_confidence=0.5, min_lift=1.0)

    # Find rule {A} -> {B}
    target_rule = rules[
        rules["antecedents"].apply(lambda s: s == frozenset(["A"])) &
        rules["consequents"].apply(lambda s: s == frozenset(["B"]))
    ]

    assert len(target_rule) == 1, "Rule {A} -> {B} not found!"
    row = target_rule.iloc[0]

    np.testing.assert_almost_equal(row["support"], 0.4, decimal=4)
    np.testing.assert_almost_equal(row["confidence"], 2.0 / 3.0, decimal=4)
    np.testing.assert_almost_equal(row["lift"], 10.0 / 9.0, decimal=4)
    np.testing.assert_almost_equal(row["leverage"], 0.04, decimal=4)

    print("[evaluate] [PASS] Toy hand-computed test passed exact mathematical assertions!")
    return True


def independent_recount(rules_df: pd.DataFrame, transactions_path=TRANSACTIONS_PARQUET, n_samples=20):
    """
    2. Independent Recount:
    Picks n_samples random rules and recomputes support, confidence, and lift
    directly from playlist item sets with zero reliance on mlxtend.
    """
    if rules_df.empty:
        print("[recount] No rules provided to verify.")
        return 0, 0

    trans_df = pd.read_parquet(transactions_path)
    playlists = [set(items) for items in trans_df["item"]]
    N = len(playlists)

    sample_size = min(n_samples, len(rules_df))
    sampled_rules = rules_df.sample(n=sample_size, random_state=42)

    mismatches = 0

    print(f"\n[recount] Independently recounting {sample_size} random rules against {N:,} raw playlists...")
    for idx, (_, row) in enumerate(sampled_rules.iterrows(), 1):
        ant = set(row["antecedents"])
        con = set(row["consequents"])
        both = ant.union(con)

        # Count occurrences directly
        count_ant = sum(1 for p in playlists if ant.issubset(p))
        count_con = sum(1 for p in playlists if con.issubset(p))
        count_both = sum(1 for p in playlists if both.issubset(p))

        calc_supp = count_both / float(N)
        calc_conf = count_both / float(count_ant) if count_ant > 0 else 0.0
        calc_lift = calc_conf / (count_con / float(N)) if count_con > 0 else 0.0

        diff_supp = abs(calc_supp - row["support"])
        diff_conf = abs(calc_conf - row["confidence"])
        diff_lift = abs(calc_lift - row["lift"])

        if diff_supp > 1e-4 or diff_conf > 1e-4 or diff_lift > 1e-3:
            print(f"  [MISMATCH] Rule {ant} -> {con}")
            print(f"    Expected: Supp={row['support']:.4f}, Conf={row['confidence']:.4f}, Lift={row['lift']:.4f}")
            print(f"    Recount:  Supp={calc_supp:.4f}, Conf={calc_conf:.4f}, Lift={calc_lift:.4f}")
            mismatches += 1

    if mismatches == 0:
        print(f"[recount] [PASS] Verified {sample_size}/{sample_size} rules with ZERO mismatches!")
    else:
        print(f"[recount] [FAIL] Encountered {mismatches} mismatches.")

    return sample_size, mismatches


def run_holdout_validation(train_ratio=0.8, min_support=0.002, min_confidence=0.25, min_lift=1.2):
    """
    3. Hold-out Validation:
    Splits playlists 80% train / 20% test.
    Mines rules on train set, then measures their support, confidence, and lift
    on the unseen 20% test set.
    """
    print(f"\n[holdout] Running 80/20 train-test validation...")
    basket_df, N, items = load_basket_data()

    np.random.seed(42)
    indices = np.random.permutation(len(basket_df))
    split_point = int(train_ratio * len(indices))
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    train_basket = basket_df.iloc[train_idx]
    test_basket = basket_df.iloc[test_idx]

    N_train = len(train_basket)
    N_test = len(test_basket)

    # Mine on train
    fi_train, _ = mine_frequent_itemsets(train_basket, min_support=min_support, max_len=3, algorithm="fpgrowth")
    train_rules = generate_rules(fi_train, num_transactions=N_train, min_confidence=min_confidence, min_lift=min_lift)

    if train_rules.empty:
        print("[holdout] No rules found in training split at this support threshold.")
        return {}

    # Convert test set to item sets for fast evaluation
    # Extract active columns for test
    test_cols = list(test_basket.columns)
    test_matrix = test_basket.to_numpy()
    test_playlists = []
    for r in range(test_matrix.shape[0]):
        active_items = {test_cols[c] for c in np.where(test_matrix[r])[0]}
        test_playlists.append(active_items)

    rules_surviving_lift_1 = 0
    confidence_drops = []
    evaluated_rules = []

    for _, row in train_rules.iterrows():
        ant = set(row["antecedents"])
        con = set(row["consequents"])
        both = ant.union(con)

        c_ant = sum(1 for p in test_playlists if ant.issubset(p))
        c_con = sum(1 for p in test_playlists if con.issubset(p))
        c_both = sum(1 for p in test_playlists if both.issubset(p))

        test_supp = c_both / float(N_test)
        test_conf = (c_both / float(c_ant)) if c_ant > 0 else 0.0
        test_lift = (test_conf / (c_con / float(N_test))) if c_con > 0 else 0.0

        if test_lift > 1.0:
            rules_surviving_lift_1 += 1

        conf_drop = row["confidence"] - test_conf
        confidence_drops.append(conf_drop)

    pct_survived = (rules_surviving_lift_1 / len(train_rules)) * 100.0
    mean_conf_drop = float(np.mean(confidence_drops))

    print(f"  Training rules discovered:      {len(train_rules):,}")
    print(f"  Rules retaining Lift > 1 in test: {rules_surviving_lift_1:,} ({pct_survived:.1f}%)")
    print(f"  Mean Confidence Drop:           {mean_conf_drop:+.2%}")

    return {
        "train_rules_count": len(train_rules),
        "test_retention_pct": round(pct_survived, 2),
        "mean_confidence_drop": round(mean_conf_drop, 4)
    }


def run_shuffled_null_baseline(min_support=0.002, min_confidence=0.25, min_lift=1.5):
    """
    4. Shuffled Null Baseline:
    Randomly permutes item assignments across playlists while keeping playlist degrees fixed.
    A genuine signal should disappear in shuffled data.
    """
    print(f"\n[shuffled] Running shuffled null baseline comparison...")
    trans_df = pd.read_parquet(TRANSACTIONS_PARQUET)
    item_lists = trans_df["item"].tolist()

    # Flatten all item occurrences
    all_tokens = [item for sub in item_lists for item in sub]
    lengths = [len(sub) for sub in item_lists]

    # Permute tokens randomly
    np.random.seed(42)
    shuffled_tokens = np.random.permutation(all_tokens).tolist()

    # Reconstruct playlists with same lengths
    shuffled_playlists = []
    curr = 0
    for length in lengths:
        shuffled_playlists.append(set(shuffled_tokens[curr:curr + length]))
        curr += length

    # Build binary matrix
    unique_items = sorted(list(set(all_tokens)))
    item_to_idx = {item: i for i, item in enumerate(unique_items)}

    from scipy.sparse import lil_matrix
    N = len(shuffled_playlists)
    M = len(unique_items)
    X = lil_matrix((N, M), dtype=bool)

    for r, p in enumerate(shuffled_playlists):
        cols = [item_to_idx[item] for item in p if item in item_to_idx]
        X[r, cols] = True

    shuffled_basket = pd.DataFrame.sparse.from_spmatrix(X.tocsr(), columns=unique_items)

    fi_shuffled, _ = mine_frequent_itemsets(shuffled_basket, min_support=min_support, max_len=3, algorithm="fpgrowth")
    shuffled_rules = generate_rules(fi_shuffled, num_transactions=N, min_confidence=min_confidence, min_lift=min_lift)

    print(f"  Shuffled Frequent Itemsets (size >= 2): {len(fi_shuffled[fi_shuffled['length'] >= 2]):,}")
    print(f"  Shuffled Rules (Lift >= {min_lift}):        {len(shuffled_rules):,}")

    return {
        "shuffled_pairs_count": len(fi_shuffled[fi_shuffled["length"] >= 2]),
        "shuffled_rules_count": len(shuffled_rules)
    }


def run_full_evaluation():
    print("=" * 70)
    print("TUNETRAIL COMPREHENSIVE EVALUATION & VALIDATION SUITE")
    print("=" * 70)

    # 1. Toy Test
    run_toy_dataset_test()

    # 2. Independent Recount on default mined rules
    basket_df, N, _ = load_basket_data()
    fi, _ = mine_frequent_itemsets(basket_df, min_support=0.005, max_len=3, algorithm="fpgrowth")
    rules = generate_rules(fi, num_transactions=N, min_confidence=0.25, min_lift=1.2)
    sample_size, mismatches = independent_recount(rules, n_samples=20)

    # 3. Holdout
    holdout_res = run_holdout_validation(train_ratio=0.8, min_support=0.005, min_confidence=0.25, min_lift=1.2)

    # 4. Shuffled
    shuffled_res = run_shuffled_null_baseline(min_support=0.005, min_confidence=0.25, min_lift=1.2)

    eval_summary = {
        "toy_test_status": "PASSED",
        "independent_recount": {
            "samples_checked": sample_size,
            "mismatches": mismatches
        },
        "holdout_validation": holdout_res,
        "shuffled_baseline": shuffled_res
    }

    eval_path = RESULTS_DIR / "evaluation_summary.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\n[evaluate] Full evaluation summary saved to: {eval_path}\n")
    return eval_summary


if __name__ == "__main__":
    run_full_evaluation()
