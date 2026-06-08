"""
Data loading for the StarCraft II match dataset.

CSV format (10 columns, NO header):
    date, playerA, [winner|loser], score, playerB, [winner|loser],
    raceA, raceB, version, online/offline

Key design note: the FULL dataset gives every player ~386 matches, so naive
skill estimates are easy and plain win-rate is a strong baseline. The
interesting science (cold-start, data-size, uncertainty) lives in the
LIMITED-DATA regime, which we CREATE by subsampling. That is what
subsample_matches() and fraction_sample() are for.
"""
from random import seed

import numpy as np
import pandas as pd

RACES = ["P", "T", "Z", "R"]  # Protoss, Terran, Zerg, Random


def load_matches(path):
    """Parse a match CSV into a clean DataFrame.
    Returns columns: date, winner, loser, race_w, race_l, online (bool).
    Skip rows that don't split into exactly 10 fields, and rows where the
    winner/loser flags are ambiguous.
    """
    rows = []
    with open(path) as f:
        for line in f:
            c = line.rstrip("\n").split(",")
            if len(c) != 10:
                continue
            wA = c[2] == "[winner]"
            wB = c[5] == "[winner]"
            if wA == wB:
                continue
            if wA:
                winner, loser = c[1], c[4]
                race_w, race_l = c[6], c[7]
            else:
                winner, loser = c[4], c[1]
                race_w, race_l = c[7], c[6]
            online = c[9].strip() == "online"
            rows.append((c[0], winner, loser, race_w, race_l, online))
    
    df = pd.DataFrame(rows, columns=["date", "winner", "loser", "race_w", "race_l", "online"])
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    return df
    

def build_player_index(*dfs):
    """Map every player across all given DataFrames to a contiguous int id.
    Build across train AND valid together so validation players are known.
    Returns (pid: dict name->id, names: list id->name).
    """
    names = set()
    for df in dfs:
        names.update(df["winner"].unique())
        names.update(df["loser"].unique())
    names = sorted(names)
    pid = {n: i for i, n in enumerate(names)}
    return pid, names


def as_pairs(df, pid, seed=0):
    """Return symmetric (a, b, outcome) arrays with randomized orientation.
    outcome = 1 if a beat b, else 0. Randomizing the orientation stops a
    model from cheating off a fixed winner-first ordering.
    """
    rng = np.random.default_rng(seed)
    w = df["winner"].map(pid).values
    l = df["loser"].map(pid).values
    flip = rng.random(len(df)) < 0.5
    a = np.where(flip, w, l)     # where flip: loser goes first
    b = np.where(flip, l, w)
    outcome = np.where(flip, 1, 0).astype(int)   # flip -> a(loser) lost -> 0
    return a, b, outcome


def subsample_matches(df, target_player, n_keep, seed=0):
    """Cold-start helper: keep only n_keep of target_player's matches,
    plus ALL matches among everyone else. Simulates a 'new' player."""
    rng = np.random.default_rng(seed)
    involves = (df["winner"] == target_player) | (df["loser"] == target_player)
    target_rows = df.index[involves].to_numpy()
    rng.shuffle(target_rows)
    keep_target = set(target_rows[:n_keep])
    keep = (~involves) | df.index.isin(keep_target)
    return df[keep].reset_index(drop=True)



def fraction_sample(df, frac, seed=0):
    """Keep a random `frac` of all matches (data-size experiment)."""
    return df.sample(frac=frac, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    # Once implemented, this should print ~192k train / ~94k valid, 999 players.
    tr = load_matches("data/train.csv")
    va = load_matches("data/valid.csv")
    pid, names = build_player_index(tr, va)
    print(f"train matches: {len(tr):,}   valid matches: {len(va):,}")
    print(f"unique players: {len(names):,}")
