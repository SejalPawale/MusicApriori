"""
03_mine_test.py: Step 3 quick CLI test and independent recount validation.

Usage:
  python 03_mine_test.py [min_support]
Example:
  python 03_mine_test.py 0.002
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mining import load_basket_data, mine_frequent_itemsets, generate_rules
from rules import filter_rules, generate_rule_explanation
from evaluate import independent_recount


def main():
    min_support = 0.002
    if len(sys.argv) > 1:
        try:
            min_support = float(sys.argv[1])
        except ValueError:
            print(f"Invalid min_support argument '{sys.argv[1]}', using default 0.002")

    min_confidence = 0.25
    min_lift = 1.2
    max_len = 3

    print("=" * 70)
    print(f"TUNETRAIL MINING TEST (min_support={min_support:.4%})")
    print("=" * 70)

    try:
        basket_df, num_transactions, items = load_basket_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 02_preprocess.py first!")
        return

    print(f"Transactions: {num_transactions:,} | Items: {len(items):,}")
    print(f"Mining frequent itemsets (max_len={max_len})...")

    fi, fi_time = mine_frequent_itemsets(basket_df, min_support=min_support, max_len=max_len, algorithm="fpgrowth")

    print(f"Elapsed: {fi_time:.3f}s")
    print("\nFrequent Itemsets by Size:")
    if fi.empty:
        print("  None found! Consider lowering min_support (e.g. python 03_mine_test.py 0.001)")
        return

    size_counts = fi["length"].value_counts().sort_index()
    for sz, count in size_counts.items():
        print(f"  Size {sz}: {count:,} itemsets")

    print(f"\nGenerating association rules (min_confidence={min_confidence}, min_lift={min_lift})...")
    rules = generate_rules(fi, num_transactions, min_confidence=min_confidence, min_lift=min_lift)
    print(f"Total Rules Discovered: {len(rules):,}")

    if rules.empty:
        print("  No rules passed the confidence/lift thresholds. Try lowering thresholds.")
        return

    # Add explanations
    rules["explanation"] = rules.apply(generate_rule_explanation, axis=1)

    print("\n" + "-" * 70)
    print("TOP 10 ASSOCIATION RULES BY LIFT:")
    print("-" * 70)
    top_10 = rules.head(10)
    for rank, (_, row) in enumerate(top_10.iterrows(), 1):
        ant = row["antecedents_str"]
        con = row["consequents_str"]
        print(f"\n[{rank}] {ant}  ➔  {con}")
        print(f"    Support: {row['support']:.4%} | Conf: {row['confidence']:.2%} | Lift: {row['lift']:.2f} | Leverage: {row['leverage']:.4f}")
        print(f"    Explanation: {row['explanation']}")

    # Independent recount check
    print("\n" + "=" * 70)
    print("INDEPENDENT RECOUNT VERIFICATION (20 RANDOM RULES):")
    print("=" * 70)
    samples_checked, mismatches = independent_recount(rules, n_samples=20)
    if mismatches == 0:
        print(f"\nSUCCESS: Recount verified on {samples_checked} rules with 0 mismatches!")
    else:
        print(f"\nWARNING: Found {mismatches} mismatches.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
