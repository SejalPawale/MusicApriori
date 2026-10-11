"""
viz.py: Visualizations for Association Rule Mining.

Implements the four required visualizations:
1. Frequent itemsets by size (bar chart)
2. Top rules by lift or confidence (horizontal bar chart)
3. Support vs Confidence scatter plot colored by Lift (Plotly)
4. NetworkX graph of rules with node size = playlist count and edge width = lift
+ Bonus: Ego network graph for individual song/artist search
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def plot_itemsets_by_size(frequent_itemsets: pd.DataFrame):
    """
    Visualization 1: Bar chart of frequent itemset counts grouped by itemset size (1, 2, 3...).
    """
    if frequent_itemsets.empty:
        fig = px.bar(title="No Frequent Itemsets Available")
        return fig

    fi = frequent_itemsets.copy()
    if "length" not in fi.columns:
        fi["length"] = fi["itemsets"].apply(len)

    counts = fi["length"].value_counts().sort_index().reset_index()
    counts.columns = ["Itemset Size (k)", "Count"]

    fig = px.bar(
        counts,
        x="Itemset Size (k)",
        y="Count",
        text="Count",
        title="Frequent Itemset Distribution by Size (k-itemsets)",
    )
    fig.update_traces(
        marker_color="#2b3e50",
        marker_line_color="#1a252f",
        marker_line_width=1,
        textposition="outside"
    )
    fig.update_layout(
        xaxis=dict(tickmode="linear", tick0=1, dtick=1, title="Itemset Size k (1 = Single Track, 2 = Pair, 3 = Triplet)"),
        yaxis=dict(title="Number of Frequent Itemsets"),
        template="simple_white",
        font=dict(family="Arial, sans-serif", size=12, color="#222222"),
        margin=dict(l=60, r=40, t=60, b=50)
    )
    return fig


def plot_top_rules_bar(rules_df: pd.DataFrame, metric: str = "lift", top_n: int = 15):
    """
    Visualization 2: Horizontal bar chart of top rules ranked by metric (lift or confidence).
    Clean, informative, monochrome/slate theme.
    """
    if rules_df.empty:
        fig = px.bar(title="No Rules to Display")
        return fig

    top_rules = rules_df.sort_values(by=metric, ascending=True).tail(top_n).copy()

    def make_short_label(row):
        a = row.get("antecedents_str", " + ".join(list(row["antecedents"])))
        c = row.get("consequents_str", " + ".join(list(row["consequents"])))
        if len(a) > 28:
            a = a[:25] + "..."
        if len(c) > 28:
            c = c[:25] + "..."
        return f"{a}  ➔  {c}"

    top_rules["Rule"] = top_rules.apply(make_short_label, axis=1)

    fig = px.bar(
        top_rules,
        x=metric,
        y="Rule",
        orientation="h",
        text=metric,
        title=f"Top {top_n} Association Rules Ranked by {metric.capitalize()}",
    )
    fig.update_traces(
        marker_color="#34495e",
        marker_line_color="#2c3e50",
        marker_line_width=1,
        texttemplate="%{text:.2f}×",
        textposition="outside"
    )
    fig.update_layout(
        template="simple_white",
        yaxis_title="",
        xaxis_title=f"{metric.capitalize()} Ratio (Times More Frequent Than Random Chance)",
        font=dict(family="Arial, sans-serif", size=12, color="#222222"),
        margin=dict(l=40, r=50, t=60, b=50)
    )
    return fig


def plot_support_vs_confidence(rules_df: pd.DataFrame):
    """
    Visualization 3: Scatter plot of Support vs Confidence with clean Greys/Slate Lift scaling.
    Informative labels and academic layout.
    """
    if rules_df.empty:
        fig = px.scatter(title="No Rules to Display")
        return fig

    df = rules_df.copy()
    if "antecedents_str" not in df.columns:
        df["antecedents_str"] = df["antecedents"].apply(lambda s: " + ".join(list(s)))
    if "consequents_str" not in df.columns:
        df["consequents_str"] = df["consequents"].apply(lambda s: " + ".join(list(s)))

    df["support_pct"] = df["support"] * 100.0
    df["confidence_pct"] = df["confidence"] * 100.0

    fig = px.scatter(
        df,
        x="support_pct",
        y="confidence_pct",
        color="lift",
        color_continuous_scale="Blues",
        hover_name="antecedents_str",
        hover_data={
            "consequents_str": True,
            "support_pct": ":.3f",
            "confidence_pct": ":.1f",
            "lift": ":.2f",
            "co_occurrence_count": True if "co_occurrence_count" in df.columns else False
        },
        labels={
            "support_pct": "Support (% of all playlists)",
            "confidence_pct": "Confidence (%)",
            "lift": "Lift"
        },
        title="Rule Support vs. Confidence (Color Shade = Lift)"
    )

    fig.update_traces(
        marker=dict(size=8, opacity=0.85, line=dict(width=0.8, color="#1c2833"))
    )

    fig.update_layout(
        template="simple_white",
        coloraxis_colorbar=dict(
            title=dict(text="Lift", side="top"),
            thickness=14,
            len=0.75
        ),
        font=dict(family="Arial, sans-serif", size=12, color="#222222"),
        margin=dict(l=60, r=40, t=60, b=50)
    )
    return fig


def plot_network_graph(rules_df: pd.DataFrame, item_meta_df: pd.DataFrame = None, top_n: int = 50):
    """
    Visualization 4: NetworkX interactive graph of top association rules.
    - Nodes: Tracks or Artists
    - Node size: Frequency / playlist count
    - Edges: Directed association (Antecedent -> Consequent)
    - Edge width: Proportional to Lift
    """
    if rules_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Rules for Network Graph")
        return fig

    sub_rules = rules_df.head(top_n).copy()
    G = nx.DiGraph()

    # Pre-build item count lookup if meta provided
    item_counts = {}
    if item_meta_df is not None and not item_meta_df.empty:
        item_counts = dict(zip(item_meta_df["item"], item_meta_df["playlist_count"]))

    # Add edges
    for _, row in sub_rules.iterrows():
        antecedents = list(row["antecedents"])
        consequents = list(row["consequents"])
        lift_val = float(row["lift"])
        conf_val = float(row["confidence"])

        for a in antecedents:
            for c in consequents:
                G.add_edge(a, c, weight=lift_val, confidence=conf_val)

    if len(G.nodes) == 0:
        fig = go.Figure()
        return fig

    # Compute layout using spring layout with k distance
    pos = nx.spring_layout(G, k=0.6 / np.sqrt(len(G.nodes)), iterations=50, seed=42)

    # Edge traces
    edge_x = []
    edge_y = []
    edge_hover_text = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_hover_text.append(f"{u} ➔ {v}<br>Lift: {data['weight']:.2f} | Conf: {data['confidence']:.1%}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="#888"),
        hoverinfo="none",
        mode="lines"
    )

    # Node traces
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []

    degrees = dict(G.degree())
    max_count = max(item_counts.values()) if item_counts else 100

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        count = item_counts.get(node, degrees.get(node, 1) * 10)
        # Scale node size between 12 and 36
        scaled_size = 12 + 24 * np.sqrt(count / max_count)
        node_size.append(scaled_size)
        node_color.append(degrees.get(node, 1))

        # Truncate long names for hover
        display_label = f"<b>{node}</b><br>Degrees: {degrees.get(node, 0)}<br>Playlists: {count:,}"
        node_text.append(display_label)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[n if len(n) <= 20 else n[:18] + ".." for n in G.nodes()],
        textposition="top center",
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale="Teal",
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title=dict(text="Connections", side="right"),
                xanchor="left"
            ),
            line_width=2
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=f"Association Rule Network (Top {len(sub_rules)} Rules)", font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=10, r=10, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white"
        )
    )
    return fig


def plot_ego_network(rules_df: pd.DataFrame, target_item: str, top_n: int = 25):
    """
    Builds and renders an ego network centered on target_item.
    Shows all items associated with target_item either as antecedent or consequent.
    """
    if rules_df.empty or not target_item:
        fig = go.Figure()
        fig.update_layout(title="No Rules Available")
        return fig

    # Filter rules containing target_item
    def has_target(row):
        return (target_item in row["antecedents"]) or (target_item in row["consequents"])

    ego_rules = rules_df[rules_df.apply(has_target, axis=1)].head(top_n)

    if ego_rules.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No association rules found involving '{target_item}'")
        return fig

    G = nx.DiGraph()
    for _, row in ego_rules.iterrows():
        a = " + ".join(list(row["antecedents"]))
        c = " + ".join(list(row["consequents"]))
        G.add_edge(a, c, lift=row["lift"], conf=row["confidence"])

    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color="#aaa"), mode="lines", hoverinfo="none")

    node_x, node_y, node_colors, node_sizes, node_text = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        is_center = (node == target_item)
        node_colors.append("#e74c3c" if is_center else "#3498db")
        node_sizes.append(28 if is_center else 16)
        node_text.append(f"<b>{'[SELECTED] ' if is_center else ''}{node}</b>")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[n if len(n) <= 22 else n[:20] + ".." for n in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(color=node_colors, size=node_sizes, line_width=2)
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Ego Network for '{target_item}' ({len(ego_rules)} Rules)",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=10, r=10, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white"
        )
    )
    return fig


def export_all_visualizations(min_support=0.005, min_confidence=0.25, min_lift=1.2):
    """
    CLI export runner: mines rules and saves all 4 required visualizations to results/ directory.
    """
    from config import RESULTS_DIR, ITEM_META_PARQUET
    from mining import load_basket_data, mine_frequent_itemsets, generate_rules
    from rules import filter_rules

    print("=" * 70)
    print("TUNETRAIL: EXPORTING ALL 4 REQUIRED VISUALIZATIONS")
    print("=" * 70)

    basket_df, N, items = load_basket_data()
    print(f"Loaded basket with {N:,} transactions across {len(items):,} items.")

    item_meta_df = pd.read_parquet(ITEM_META_PARQUET) if ITEM_META_PARQUET.exists() else None

    # 1. Mine itemsets & rules
    print(f"Mining frequent itemsets at min_support={min_support:.4%}...")
    fi, _ = mine_frequent_itemsets(basket_df, min_support=min_support, max_len=3, algorithm="fpgrowth")
    print(f"Generating rules at min_confidence={min_confidence}, min_lift={min_lift}...")
    rules = generate_rules(fi, num_transactions=N, min_confidence=min_confidence, min_lift=min_lift)
    rules = filter_rules(rules, hide_same_artist=False, min_co_occurrence=3)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Visualization 1: Itemsets by size
    print("[1/4] Exporting Frequent Itemsets by Size bar chart...")
    fig1 = plot_itemsets_by_size(fi)
    path1 = RESULTS_DIR / "viz_1_itemsets_by_size.html"
    fig1.write_html(str(path1))
    print(f"      Saved to: {path1.name}")

    # Visualization 2: Top rules bar chart
    print("[2/4] Exporting Top Rules by Lift horizontal bar chart...")
    fig2 = plot_top_rules_bar(rules, metric="lift", top_n=15)
    path2 = RESULTS_DIR / "viz_2_top_rules_bar.html"
    fig2.write_html(str(path2))
    print(f"      Saved to: {path2.name}")

    # Visualization 3: Support vs Confidence scatter
    print("[3/4] Exporting Support vs Confidence scatter plot...")
    fig3 = plot_support_vs_confidence(rules)
    path3 = RESULTS_DIR / "viz_3_support_vs_confidence.html"
    fig3.write_html(str(path3))
    print(f"      Saved to: {path3.name}")

    # Visualization 4: Network Graph (Interactive Plotly + Static PNG)
    print("[4/4] Exporting Association Rules Network Graph...")
    fig4 = plot_network_graph(rules, item_meta_df, top_n=40)
    path4 = RESULTS_DIR / "viz_4_rules_network.html"
    fig4.write_html(str(path4))
    print(f"      Saved to: {path4.name}")

    # Also generate a static Matplotlib Network Graph PNG for easy report embedding
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        G = nx.DiGraph()
        top_subset = rules.head(25)
        for _, row in top_subset.iterrows():
            a = list(row["antecedents"])[0] if len(row["antecedents"]) > 0 else "A"
            c = list(row["consequents"])[0] if len(row["consequents"]) > 0 else "C"
            G.add_edge(a, c, weight=row["lift"])

        pos = nx.spring_layout(G, k=0.8, seed=42)
        plt.style.use("default")
        plt.figure(figsize=(11, 8.5))
        nx.draw_networkx_nodes(
            G, pos,
            node_color="#2c3e50",
            node_size=1100,
            alpha=0.9
        )
        nx.draw_networkx_edges(
            G, pos,
            edge_color="#7f8c8d",
            width=1.2,
            arrowstyle="->",
            arrowsize=12
        )
        # Clean labels
        short_labels = {n: n[:17] + ".." if len(n) > 19 else n for n in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels=short_labels,
            font_size=8,
            font_family="sans-serif",
            font_color="#ffffff",
            font_weight="bold"
        )
        plt.title("Co-Occurrence Association Rule Network (Spotify Curated Playlists)", fontsize=13, fontweight="bold", pad=15)
        plt.axis("off")
        plt.tight_layout()
        path4_png = RESULTS_DIR / "viz_4_rules_network.png"
        plt.savefig(path4_png, dpi=300, facecolor="#ffffff")
        plt.close()
        print(f"      Saved static image to: {path4_png.name}")
    except Exception as e:
        print(f"      (Matplotlib PNG export notice: {e})")

    print("\nAll 4 visualizations exported successfully!")
    print("=" * 70)


if __name__ == "__main__":
    export_all_visualizations()

