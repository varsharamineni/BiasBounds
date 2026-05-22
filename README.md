# Fairness Experiments — Standalone Package

## Requirements
```
pip install numpy pandas matplotlib scipy seaborn scikit-learn tqdm gurobipy
```
gurobipy is optional (used for LP bounds). If not installed, analytic fallback is used.

## Structure
```
fairness_experiments/
├── core/
│   ├── baseline.py          Kallus bounds + tight Fréchet bounds
│   ├── data_generation.py   Simulated scenario generation
│   ├── joint_solver.py      Feasible set enumeration
│   ├── metrics.py           DD, DI and subgroup metrics
│   └── scenarios/
│       └── scenario_a.py    Scenario A dataclass + generator
├── data/
│   ├── loader.py            Real data CSV loader
│   └── real/                ← put adult.csv, compas.csv, german.csv here
├── experiments/
│   └── run_experiment.py    Main experiment runner
├── analysis/
│   ├── paper_plots.py       Figures 1–3 + real world figures
│   ├── paper_tables.py      All tables + paper text numbers
│   ├── fig_real_datasets.py Per-dataset real world figures
│   ├── fig_distribution_vs_baseline.py
│   ├── fig_mean_vs_midpoint.py
│   ├── fig_bounds_comparison.py
│   └── fig_comprehensive_comparison.py
├── figures/                 Pre-generated paper figures (PNGs)
└── results/                 Pre-computed table CSVs
```

## Quick test (~30 sec)
```bash
cd fairness_experiments
python experiments/run_experiment.py --n_gt 3 --num_grid 15 --output results/test.csv
python analysis/paper_tables.py --results results/test.csv
python analysis/paper_plots.py  --simulated results/test.csv --figures figures/test --bounds_n_gt 3 --bounds_grid 15
```

## Full run
```bash
cd fairness_experiments

# Step 1: simulated experiment (~45 min, 25k scenarios)
python experiments/run_experiment.py \
    --n_gt 3572 --num_grid 50 --seed 42 \
    --output results/experiment_results_25k.csv

# Step 2: all figures
python analysis/paper_plots.py \
    --simulated results/experiment_results_25k.csv \
    --figures   figures

# Step 3: real world figures (put CSVs in data/real/ first)
python analysis/fig_real_datasets.py \
    --data_dir data/real \
    --figures  figures

# Step 4: all tables + paper text numbers
python analysis/paper_tables.py \
    --results results/experiment_results_25k.csv \
    --out     results
```

## Real data column config (data/loader.py DATASET_CONFIGS)
Adult:  a=race  z=marital-status  x=workclass       y=income
COMPAS: a=race  z=relationship    x=capital-gain     y=income
German: a=gender z=marital-status x=workclass        y=income
