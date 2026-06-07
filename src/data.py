"""
Data loading for the StarCraft II match dataset.   [PHASE 1 - owner: P1]

CSV format (10 columns, NO header):
    date, playerA, [winner|loser], score, playerB, [winner|loser],
    raceA, raceB, version, online/offline

Key design note: the FULL dataset gives every player ~386 matches, so naive
skill estimates are easy and plain win-rate is a strong baseline. The
interesting science (cold-start, data-size, uncertainty) lives in the
LIMITED-DATA regime, which we CREATE by subsampling. That is what
subsample_matches() and fraction_sample() are for.

Implement each function below. Delete the NotImplementedError once done.
See the build guide, Phase 1, for the parsing logic and "done when" check.
"""
import numpy as np
import pandas as pd

RACES = ["P", "T", "Z", "R"]  # Protoss, Terran, Zerg, Random


def load_matches(path):
    """Parse a match CSV into a clean DataFrame.

    Returns columns: date, winner, loser, race_w, race_l, online (bool).
    Skip rows that don't split into exactly 10 fields, and rows where the
    winner/loser flags are ambiguous.
    """
    raise NotImplementedError("Phase 1 - P1")


def build_player_index(*dfs):
    """Map every player across all given DataFrames to a contiguous int id.

    Build across train AND valid together so validation players are known.
    Returns (pid: dict name->id, names: list id->name).
    """
    raise NotImplementedError("Phase 1 - P1")


def as_pairs(df, pid, seed=0):
    """Return symmetric (a, b, outcome) arrays with randomized orientation.

    outcome = 1 if a beat b, else 0. Randomizing the orientation stops a
    model from cheating off a fixed winner-first ordering.
    """
    raise NotImplementedError("Phase 1 - P1")


def subsample_matches(df, target_player, n_keep, seed=0):
    """Cold-start helper: keep only n_keep of target_player's matches,
    plus ALL matches among everyone else. Simulates a 'new' player."""
    raise NotImplementedError("Phase 1 - P1")


def fraction_sample(df, frac, seed=0):
    """Keep a random `frac` of all matches (data-size experiment)."""
    raise NotImplementedError("Phase 1 - P1")


if __name__ == "__main__":
    # Once implemented, this should print ~192k train / ~94k valid, 999 players.
    tr = load_matches("data/train.csv")
    va = load_matches("data/valid.csv")
    pid, names = build_player_index(tr, va)
    print(f"train matches: {len(tr):,}   valid matches: {len(va):,}")
    print(f"unique players: {len(names):,}")
