# Bayesian Player Skill Estimation — Graphical Models Project

Latent Bayesian skill models for predicting StarCraft II match outcomes, with a
focus on **uncertainty** and the **limited-data regime**. Course project; final
deliverable is a 4-page PDF report plus a separate code upload.

## Why this design (read this first)

The full dataset has 999 players averaging ~386 matches each — nobody has fewer
than 33 games. So naive skill estimates are easy and plain win-rate is a strong
baseline. Per the professor's note, the interesting science lives in the
**limited-data setting**, which we **create by subsampling** (hiding most of a
player's games). That is the whole point of the cold-start and data-size
experiments — there are no naturally "new" players in the data.

## Setup

```bash
git clone <this-repo-url>
cd skillproj
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import pymc; print(pymc.__version__)"   # sanity check
```

The dataset (`data/train.csv`, `data/valid.csv`) is already in the repo.

## The plan in one line

Elo baseline → Bayesian Bradley-Terry (the graphical model) → uncertainty
analysis (data-size, cold-start, calibration) → race-effect extension.

## Models

| File | What it is | Role |
|------|-----------|------|
| `src/elo.py` | Online point-estimate rating, no uncertainty | Comparison baseline |
| `src/bt_model.py` | Bayesian Bradley-Terry: `skill_i ~ N(0,σ)`, `P(a>b)=sigmoid(skill_a−skill_b)`, `outcome~Bernoulli`. Full posterior per player. | Main graphical model |
| `src/bt_model.py` (race) | Adds a learned race-matchup term to the logit | Extension |

## Experiments (each one = a report figure)

```bash
python src/experiments.py elo          # baseline number
python src/experiments.py compare      # Elo vs Bayesian on full data
python src/experiments.py datasize     # accuracy vs 5/10/25/50/100% data
python src/experiments.py coldstart    # one player's posterior as games are revealed
python src/experiments.py calibration  # reliability diagram + ECE
python src/experiments.py race         # learned race-matchup effects
```

Outputs (`.json` + `.png`) land in `results/`.

## Work split (3 people)

| Person | Owns | Phases |
|--------|------|--------|
| **P1** | Data pipeline, metrics, Elo baseline, EDA | 1–3 |
| **P2** | Bayesian model + inference (+ race extension) | 4 |
| **P3** | The five experiments, plots, report assembly | 5–6 |

Phase 0 (environment) is everyone. See `BUILD_GUIDE` (shared separately) for the
full phase-by-phase instructions, code snippets, and "done when" checks.

## Repo layout

```
data/       train.csv  valid.csv          (provided)
src/        data.py  metrics.py  elo.py  bt_model.py  experiments.py   (skeletons to fill in)
results/    generated json + png          (gitignored except .gitkeep)
report/     the 4-page PDF goes here
```

## Status

All `src/` files are **skeletons** — function signatures and docstrings with
`NotImplementedError`. Fill them in per the build guide. Nothing is implemented
yet; that's the team's work.

## Dataset

Professional StarCraft II match results from Aligulac, 2010–2017: ~286k matches
across 999 players. Columns (no header): date, playerA, win/lose flag, score,
playerB, win/lose flag, raceA, raceB, game version, online/offline.
