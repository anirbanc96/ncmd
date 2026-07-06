# NCMD

This repository contains codes for experiments from the paper: Conditional Mean Independence and Global Sensitivity Analysis using Nearest Neighbor Graphs.

---

## Repository Structure

```
ncmd/
├── Mean Independence Test/
│   ├── tests.py             # Test implementations
│   ├── utils.py             # Data generation & experiment runners
│   ├── plot_results.py      # Plotting utilities
│   ├── Univariate.ipynb     # Univariate simulation notebook
│   ├── Multivariate.ipynb   # Multivariate simulation notebook
│   └── results/             # Saved power curve and timing PDFs
│
├── Variable Importance/
│   ├── Model.py             # Sobol index estimation via NCMD
│   ├── Over_n.py            # Convergence experiment over sample size
│   ├── Over_Theta.py        # Sensitivity experiment over theta
│   └── *.pdf                # Output figures
│
└── Variable Screening/
    ├── utils.py                     # Screening methods & simulation runners
    ├── simulations.ipynb            # Simulation experiments
    ├── california_housing_01.ipynb  # Real-data experiment (sigma = 0.1)
    ├── california_housing_05.ipynb  # Real-data experiment (sigma = 0.5)
    └── california_housing_1.ipynb   # Real-data experiment (sigma = 1)
```

---

## Experiments

### Mean Independence Test

Benchmarks several tests for the null hypothesis H₀: E[Y | X] = E[Y] across univariate and multivariate settings with varying noise levels (λ). Compares:

- **NCMD** — nearest-neighbor conditional mean independence test (proposed)
- **Chatterjee** — rank-based independence test [Chatterjee, 2021]
- **Azadkia–Chatterjee** — nearest-neighbor graph coefficient of conditional dependence [Azadkia & Chatterjee, 2021]
- **MDD** — martingale difference divergence test [Shao & Zhang, 2014]
- **dCov** — distance covariance test [Székely, Rizzo & Bakirov, 2007]
- **pMIT** — split-sample prediction-based mean independence test using XGBoost [Cai, Guo & Zhong, 2025]


Outputs: power curves and timing tables across Normal, Uniform, and Beta covariate distributions, in both univariate and multivariate (d = 5) regimes.

### Variable Importance

Estimates and validates **Sobol sensitivity indices** via nearest-neighbors on a synthetic model with main effects, two-way, and three-way interactions. Experiments sweep over:

- **Sample size n** (`Over_n.py`) — convergence of estimators to population values
- **Interaction parameter θ** (`Over_Theta.py`) — sensitivity of Sobol indices as θ varies

### Variable Screening

Greedy forward variable selection using NCMD statistics, benchmarked against marginal screening methods (MDCSIS, BcorSIS, Kfilter from the R `MFSIS` package). Experiments include:

- **Simulations** (`simulations.ipynb`) — exact recovery and containment rates across six nonlinear data-generating processes, varying n and p
- **California Housing** (`california_housing_*.ipynb`) — real-data feature selection at three noise thresholds ($\sigma$ ∈ {0.1, 0.5, 1})
