# ncmd

A collection of nonparametric statistical experiments based on nearest-neighbor conditional mean dependence (NNCMD). The repository contains three self-contained experiment directories.

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
│   ├── Model.py             # Sobol index estimation via NNCMD
│   ├── Over_n.py            # Convergence experiment over sample size
│   ├── Over_Theta.py        # Sensitivity experiment over theta
│   └── *.pdf                # Output figures
│
└── Variable Screening/
    ├── utils.py                     # Screening methods & simulation runners
    ├── simulations.ipynb            # Simulation experiments
    ├── california_housing_01.ipynb  # Real-data experiment (α = 0.01)
    ├── california_housing_05.ipynb  # Real-data experiment (α = 0.05)
    └── california_housing_1.ipynb   # Real-data experiment (α = 0.1)
```

---

## Experiments

### Mean Independence Test

Benchmarks several tests for the null hypothesis H₀: E[Y | X] = E[Y] across univariate and multivariate settings with varying noise levels (λ). Compares:

- **NNCMI** — nearest-neighbor conditional mean independence test (proposed)
- **Chatterjee** — rank-based ξ correlation test
- **Azadkia–Chatterjee** — permutation test using the T statistic
- **MDD** — martingale difference divergence with multiplier bootstrap
- **dCov** — distance covariance permutation test
- **pMIT** — XGBoost-based split test

Outputs: power curves and timing tables across Normal, Uniform, and Beta covariate distributions, in both univariate and multivariate (d = 5) regimes.

### Variable Importance

Estimates and validates **Sobol sensitivity indices** via nearest-neighbor regression on a synthetic model with main effects, two-way, and three-way interactions. Experiments sweep over:

- **Sample size n** (`Over_n.py`) — convergence of estimators to population values
- **Interaction parameter θ** (`Over_Theta.py`) — sensitivity of Sobol indices as θ varies

### Variable Screening

Greedy forward variable selection using NNCMD statistics, benchmarked against marginal screening methods (MDCSIS, BcorSIS, Kfilter from the R `MFSIS` package). Experiments include:

- **Simulations** (`simulations.ipynb`) — exact recovery and containment rates across six nonlinear data-generating processes, varying n and p
- **California Housing** (`california_housing_*.ipynb`) — real-data feature selection at three significance thresholds (α ∈ {0.01, 0.05, 0.1})
