"""
Experiment driver.

Each function below = ONE figure in the report. Run individually:
    python src/experiments.py elo
    python src/experiments.py compare
    python src/experiments.py datasize
    python src/experiments.py coldstart
    python src/experiments.py calibration
    python src/experiments.py race

Each should load data, fit a model (using src/elo.py or src/bt_model.py),
compute metrics (src/metrics.py), and save a .json + .png into results/.
Suggested order: elo -> compare -> calibration -> datasize -> coldstart -> race.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from data import load_matches, build_player_index, as_pairs, fraction_sample, subsample_matches
from elo import Elo
from bt_model import fit_bt, fit_bt_race, predict_proba
from metrics import evaluate, calibration_bins, expected_calibration_error
import numpy as np, json
import matplotlib
matplotlib.use("Agg")          # headless: save files, don't open windows
import matplotlib.pyplot as plt

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

def _load():
    tr = load_matches("data/train.csv")
    va = load_matches("data/valid.csv")
    pid, names = build_player_index(tr, va)
    return tr, va, pid, names

def exp_elo():
    """Elo baseline number on the validation set."""
    tr, va, pid, names = _load()
    tr = tr.sort_values("date").reset_index(drop=True)   # chronological
    a, b, o = as_pairs(tr, pid, seed=1)
    elo = Elo(len(names)).fit(a, b, o)
    va_a, va_b, va_o = as_pairs(va, pid, seed=2)
    p = elo.predict_proba(va_a, va_b)
    m = evaluate(p, va_o)
    m["ece"] = expected_calibration_error(p, va_o)
    json.dump(m, open(f"{RESULTS}/elo.json", "w"), indent=2)
    print("ELO:", m)

def exp_compare():
    """Full-data Bayesian BT vs Elo head-to-head."""
    tr, va, pid, names = _load()
    a, b, o = as_pairs(tr, pid, seed=1)
    post = fit_bt(a, b, o, len(names), method="advi", draws=500)
    va_a, va_b, va_o = as_pairs(va, pid, seed=2)
    p = predict_proba(post, va_a, va_b)
    m = evaluate(p, va_o)
    m["ece"] = expected_calibration_error(p, va_o)
    json.dump(m, open(f"{RESULTS}/bt_full.json", "w"), indent=2)
    print("BT full:", max)


def exp_datasize():
    """Train on 5/10/25/50/100% of matches; plot accuracy + mean posterior std."""
    tr, va, pid, names = _load()
    va_a, va_b, va_o = as_pairs(va, pid, seed=2)
    out = []
    for frac in [0.05, 0.1, 0.25, 0.5, 1.0]:
        sub = fraction_sample(tr, frac, seed=0)
        a, b, o = as_pairs(sub, pid, seed=1)
        post = fit_bt(a, b, o, len(names), method="advi", draws = 400)
        p = predict_proba(post, va_a, va_b)
        m = evaluate(p, va_o)
        m["frac"] = frac
        m["mean_std"] = float(post["std"].mean())
        out.append(m); print(frac, round(m["accuracy"], 4), round(m["mean_std"], 4))
    json.dump(out, open(f"{RESULTS}/datasize.json", "w"), indent=2)

    fr = [x["frac"] for x in out]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(fr, [x["accuracy"] for x in out], "o-", color="tab:blue")
    ax1.set_xlabel("fraction of training data"); ax1.set_ylabel("accuracy", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(fr, [x["mean_std"] for x in out], "s--", color="tab:red")
    ax2.set_ylabel("mean posterior std", color="tab:red")
    plt.title("Accuracy and uncertainty vs data size")
    plt.savefig(f"{RESULTS}/datasize.png", dpi=130, bbox_inches="tight")


def exp_coldstart():
    """Hide one player; reveal 0/1/3/5/10/20/50 games; track mean +/- std."""
    tr, va, pid, names = _load()
    target = "Maru"                       # well-known, plenty of games
    if target not in pid:
        target = tr["winner"].value_counts().index[0]
    ti = pid[target]
    out = []
    for nk in [0, 1, 3, 5, 10, 20, 50]:
        sub = subsample_matches(tr, target, nk, seed=0)
        a, b, o = as_pairs(sub, pid, seed=1)
        post = fit_bt(a, b, o, len(names), method="nuts", draws=500)   # NUTS her
        out.append({"n": nk, "mean": float(post["mean"][ti]),
                    "std": float(post["std"][ti])})
        print(out[-1])
    json.dump({"player": target, "curve": out},
              open(f"{RESULTS}/coldstart.json", "w"), indent=2)
    
    n = [x["n"] for x in out]
    mean = [x["mean"] for x in out]; std = [x["std"] for x in out]
    plt.figure(figsize=(6, 4))
    plt.errorbar(n, mean, yerr=std, fmt="o-", capsize=4)
    plt.xlabel("matches observed for new player"); plt.ylabel("posterior skill")
    plt.title(f"Cold-start: {target} (mean ± 1 std)")
    plt.savefig(f"{RESULTS}/coldstart.png", dpi=130, bbox_inches="tight")


def exp_calibration():
    """Reliability diagram + ECE on full-data predictions."""
    tr, va, pid, names = _load()
    a, b, o = as_pairs(tr, pid, seed=1)
    post = fit_bt(a, b, o, len(names), method="advi", draws=500)
    va_a, va_b, va_o = as_pairs(va, pid, seed=2)
    p = predict_proba(post, va_a, va_b)
    centers, obs, counts = calibration_bins(p, va_o, n_bins=10)
    ece = expected_calibration_error(p, va_o)
    json.dump({"centers": centers.tolist(), "observed": obs.tolist(),
               "counts": counts.tolist(), "ece": ece},
              open(f"{RESULTS}/calibration.json", "w"), indent=2)
    
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="perfect")
    plt.plot(centers, obs, "o-", label=f"BT (ECE={ece:.3f})")
    plt.xlabel("predicted P(win)"); plt.ylabel("observed win rate")
    plt.title("Reliability diagram"); plt.legend()
    plt.savefig(f"{RESULTS}/calibration.png", dpi=130, bbox_inches="tight")
    print("calibration ECE:", round(ece, 4))


def exp_race():
    """Fit BT+race; report learned matchup advantages."""
    tr, va, pid, names = _load()
    rmap = {"P": 0, "T": 1, "Z": 2, "R": 3}
    
    def pairs_with_race(df, seed):
        rng = np.random.default_rng(seed)
        w = df["winner"].map(pid).to_numpy(); l = df["loser"].map(pid).to_numpy()
        rw = df["race_w"].map(rmap).fillna(-1).astype(int).to_numpy()
        rl = df["race_l"].map(rmap).fillna(-1).astype(int).to_numpy()
        flip = rng.random(len(df)) < 0.5
        A = np.where(flip, l, w); B = np.where(flip, w, l)
        RA = np.where(flip, rl, rw); RB = np.where(flip, rw, rl)
        O = np.where(flip, 0, 1).astype(int)
        return A, B, RA, RB, O
    
    A, B, RA, RB, O = pairs_with_race(tr, 1)
    post = fit_bt_race(A, B, RA, RB, O, len(names), method="advi", draws=500)
    races = ["P", "T", "Z"]
    eff = {}
    for i in range(3):
        for j in range(i + 1, 3):
            k = post["pair_id"][i, j]
            eff[f"{races[i]} vs {races[j]}"] = round(float(post["race_mean"][k]), 4)
    json.dump(eff, open(f"{RESULTS}/race.json", "w"), indent=2)
    print("race effects:", eff)


CMDS = {"elo": exp_elo, "compare": exp_compare, "datasize": exp_datasize,
        "coldstart": exp_coldstart, "calibration": exp_calibration,
        "race": exp_race}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print("usage: python src/experiments.py [" + "|".join(CMDS) + "]")
    else:
        CMDS[sys.argv[1]]()
