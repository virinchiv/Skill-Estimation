"""
Experiment driver.   [PHASE 5 - owner: P3]

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
# from data import load_matches, build_player_index, as_pairs, fraction_sample, subsample_matches
# from elo import Elo
# from bt_model import fit_bt, fit_bt_race, predict_proba
# from metrics import evaluate, calibration_bins, expected_calibration_error

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


def exp_elo():
    """Elo baseline number on the validation set."""
    raise NotImplementedError("Phase 5 - P3")


def exp_compare():
    """Full-data Bayesian BT vs Elo head-to-head."""
    raise NotImplementedError("Phase 5 - P3")


def exp_datasize():
    """Train on 5/10/25/50/100% of matches; plot accuracy + mean posterior std."""
    raise NotImplementedError("Phase 5 - P3")


def exp_coldstart():
    """Hide one player; reveal 0/1/3/5/10/20/50 games; track mean +/- std."""
    raise NotImplementedError("Phase 5 - P3")


def exp_calibration():
    """Reliability diagram + ECE on full-data predictions."""
    raise NotImplementedError("Phase 5 - P3")


def exp_race():
    """Fit BT+race; report learned matchup advantages."""
    raise NotImplementedError("Phase 5 - P3 (extension)")


CMDS = {"elo": exp_elo, "compare": exp_compare, "datasize": exp_datasize,
        "coldstart": exp_coldstart, "calibration": exp_calibration,
        "race": exp_race}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print("usage: python src/experiments.py [" + "|".join(CMDS) + "]")
    else:
        CMDS[sys.argv[1]]()
