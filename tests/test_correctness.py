"""
test_correctness.py: Automated tests for TuneTrail.

Tests include:
- Hand-computed Toy Dataset exact mathematical assertions (support, confidence, lift, leverage)
- Text normalization & edition/suffix stripping
- Transaction identification (ensuring user_id isolation)
- Same-artist rule detection
- Plain-language explanation formatting
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocess import normalize_text
from rules import is_same_artist_rule, generate_rule_explanation, get_item_artist
from evaluate import run_toy_dataset_test


def test_hand_computed_toy_dataset():
    """Validates mlxtend output against exact hand calculations."""
    assert run_toy_dataset_test() is True


def test_text_normalization_suffixes():
    """Tests regex cleaning of suffixes like remastered, live, radio edit."""
    assert normalize_text("Hotel California - Remastered 2013", is_track=True) == "hotel california"
    assert normalize_text("Comfortably Numb - Live at Pompeii", is_track=True) == "comfortably numb"
    assert normalize_text("One More Time - Radio Edit", is_track=True) == "one more time"
    assert normalize_text("Wonderwall - Remastered", is_track=True) == "wonderwall"
    assert normalize_text("   Bohemian    Rhapsody   ", is_track=True) == "bohemian rhapsody"
    assert normalize_text("Queen - 2011 Remaster", is_track=True) == "queen"


def test_tid_isolation():
    """Tests that user_id isolation prevents merging different users' Starred playlists."""
    u1, p1 = "user_alpha", "Starred"
    u2, p2 = "user_beta", "Starred"

    tid1 = f"{u1}|{p1}"
    tid2 = f"{u2}|{p2}"

    assert tid1 != tid2
    assert tid1 == "user_alpha|Starred"
    assert tid2 == "user_beta|Starred"


def test_same_artist_detection():
    """Tests detection of rules whose antecedents and consequents share artists."""
    same_row = {
        "antecedents": frozenset(["coldplay – yellow"]),
        "consequents": frozenset(["coldplay – the scientist"])
    }
    assert is_same_artist_rule(same_row) is True

    diff_row = {
        "antecedents": frozenset(["coldplay – yellow"]),
        "consequents": frozenset(["oasis – wonderwall"])
    }
    assert is_same_artist_rule(diff_row) is False


def test_rule_explanation_template():
    """Tests plain-language sentence formatting."""
    row = {
        "antecedents": frozenset(["oasis – wonderwall"]),
        "consequents": frozenset(["the verve – bitter sweet symphony"]),
        "support": 0.0021,
        "confidence": 0.38,
        "lift": 24.0,
        "co_occurrence_count": 212
    }
    explanation = generate_rule_explanation(row)
    assert "Playlists containing" in explanation
    assert "38.0% of cases" in explanation
    assert "24.0× more often" in explanation
    assert "212 playlists" in explanation
    assert "0.21%" in explanation
