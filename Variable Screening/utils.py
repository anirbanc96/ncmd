import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pandas as pd
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# R / MFSIS setup
# ---------------------------------------------------------------------------

_mfsis_pkg = importr("MFSIS")

robjects.r("""
mfsis_r <- function(X, Y, method, d) {
    X <- as.matrix(X)
    Y <- as.numeric(Y)
    suppressMessages(suppressWarnings(
        result <- MFSIS(X, Y, nsis = d, method = as.character(method))
    ))
    return(result)
}
""")
_r_mfsis = robjects.globalenv["mfsis_r"]


def _mfsis_select(X, Y, method, nsis=None):
    """Generic MFSIS wrapper — calls the R MFSIS package with the given method string."""
    n, p = X.shape
    if nsis is None:
        nsis = int(n / np.log(n))
    nsis = min(nsis, p)

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1)

    with localconverter(default_converter + numpy2ri.converter):
        X_r = robjects.conversion.py2rpy(X)
        Y_r = robjects.conversion.py2rpy(Y)

    X_r = robjects.r["as.matrix"](X_r)
    result = _r_mfsis(X_r, Y_r, robjects.StrVector([method]), robjects.IntVector([nsis]))
    return [int(i) - 1 for i in result]


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def _median_bandwidth(Y):
    diffs = np.abs(Y[:, None] - Y[None, :])
    return np.median(diffs[np.triu_indices(len(Y), k=1)])


def _double_center(D):
    row_mean = D.mean(axis=1, keepdims=True)
    col_mean = D.mean(axis=0, keepdims=True)
    grand_mean = D.mean()
    return D - row_mean - col_mean + grand_mean


def compute_T_stat(X, Y, S, k_nn=10):
    """NNCMI test statistic: E[Y_i * mean_{j in NN_k(i)} Y_j]."""
    if len(S) == 0:
        return 0.0
    n = len(Y)
    X_sub = X[:, S]
    nbrs = NearestNeighbors(n_neighbors=k_nn + 1).fit(X_sub)
    _, indices = nbrs.kneighbors(X_sub)
    indices = indices[:, 1:]
    T = 0.0
    for i in range(n):
        T += np.mean(Y[i] * Y[indices[i]])
    return np.abs(T / n)


def compute_kfoci_stat(X, Y, S, k_nn=10):
    """Kernel FOCI statistic (Chatterjee). Uses Gaussian kernel on Y at NN pairs in X."""
    if len(S) == 0:
        return 0.0
    X_sub = X[:, S]
    sigma = _median_bandwidth(Y)
    if sigma <= 1e-8:
        sigma = 1.0
    nbrs = NearestNeighbors(n_neighbors=k_nn + 1).fit(X_sub)
    _, indices = nbrs.kneighbors(X_sub)
    indices = indices[:, 1:]
    Y_neighbors = Y[indices]
    diff = Y[:, None] - Y_neighbors
    K_vals = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    return np.mean(K_vals)


# ---------------------------------------------------------------------------
# Greedy forward selection
# ---------------------------------------------------------------------------

def forward_selection(X, Y, k_nn=10, statistic="nncmi"):
    """
    Greedy forward variable selection using NNCMI or Chatterjee statistic.

    Stops when no candidate variable improves the current best T value.

    Parameters
    ----------
    X         : (n, p) array
    Y         : (n,) array
    k_nn      : number of nearest neighbours
    statistic : 'nncmi' (alias 'linear') or 'chatterjee' (alias 'kfoci')

    Returns
    -------
    List of selected feature indices (0-based).
    """
    if statistic in ("nncmi", "linear"):
        T_fn = compute_T_stat
    elif statistic in ("chatterjee", "kfoci"):
        T_fn = compute_kfoci_stat
    else:
        raise ValueError(f"Unknown statistic: {statistic!r}. Use 'nncmi'/'linear' or 'chatterjee'/'kfoci'.")

    n, p = X.shape
    selected = []
    remaining = list(range(p))
    best_T = -np.inf

    while remaining:
        best_candidate = None
        best_candidate_T = best_T

        for j in remaining:
            T_val = T_fn(X, Y, selected + [j], k_nn)
            if T_val >= best_candidate_T:
                best_candidate_T = T_val
                best_candidate = j

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_T = best_candidate_T

    return selected


# ---------------------------------------------------------------------------
# Marginal screening methods
# ---------------------------------------------------------------------------

def mdc_select(X, Y, nsis=None):
    """
    Select top features by MDC-based sure independence screening.

    Default nsis = floor(n / log(n)), capped at p.
    """
    n, p = X.shape
    if nsis is None:
        nsis = int(n / np.log(n))
    nsis = min(nsis, p)

    Yc = Y - Y.mean()
    Y_outer = np.outer(Yc, Yc)
    VY = np.mean(Y_outer * Y_outer)
    scores = np.zeros(p)

    for k in range(p):
        Xk = X[:, k]
        X_dist = np.abs(Xk[:, None] - Xk[None, :])
        A = _double_center(X_dist)
        VXY = np.mean(A * Y_outer)
        VX = np.mean(A * A)
        if VX > 0 and VY > 0:
            scores[k] = VXY / np.sqrt(VX * VY)

    return list(np.argsort(-np.abs(scores))[:nsis])


def bcorsis_select(X, Y, nsis=None):
    """Ball Correlation SIS. Default nsis = floor(n / log(n)), capped at p."""
    return _mfsis_select(X, Y, "BcorSIS", nsis)


def kfilter_select(X, Y, nsis=None):
    """Kernel Filter SIS. Default nsis = floor(n / log(n)), capped at p."""
    return _mfsis_select(X, Y, "Kfilter", nsis)


# ---------------------------------------------------------------------------
# Data generation (simulation experiments)
# ---------------------------------------------------------------------------

def data_generation(n=200, p=10, setting="setting1", rho=0.7):
    """Generate synthetic (X, Y) data for simulation experiments."""
    X = np.random.normal(0, 1, (n, p))
    eps = np.random.normal(0, 1, n)
    if setting == "setting1":
        Y = X[:, 0] * X[:, 1] + X[:, 0] - X[:, 2] + eps
    elif setting == "setting2":
        Y = np.sin(X[:, 0]) + np.cos(X[:, 1]) * X[:, 2] + eps
    elif setting == "setting3":
        Y = np.where(
            X[:, 1] < 0.0,
            np.cos(X[:, 0]) + np.sin(X[:, 2]),
            np.sin(X[:, 0]) + np.cos(X[:, 2])) + eps
    elif setting == "setting4":
        Y = np.where(
            X[:, 1] < 0.0,
            np.cos(X[:, 0]) * np.exp(X[:, 2]),
            np.sin(X[:, 2]) * np.exp(X[:, 0])) + eps
    elif setting == "setting5":
        Z = np.random.normal(0, 1, (n, 1))
        E = np.random.normal(0, 1, (n, p))
        X = np.sqrt(rho) * Z + np.sqrt(1 - rho) * E
        Y = X[:, 0] ** 2 - np.sin(X[:, 1]) * np.exp(X[:, 2]) + eps
    elif setting == "setting6":
        Y = X[:, 0] * X[:, 1] + X[:, 0] - X[:, 2] + eps * (X[:, 3] + X[:, 4])
    else:
        Y = X[:, 0] * X[:, 1] + np.sin(X[:, 0] * X[:, 2]) + eps
    return X, Y


# ---------------------------------------------------------------------------
# Internal routing for simulation runners
# ---------------------------------------------------------------------------

def _select_features(X, Y, k_nn, statistic, nsis):
    """Route to the correct selection function for simulation experiments."""
    if statistic in ("linear", "nncmi"):
        return forward_selection(X, Y, k_nn, "nncmi")
    elif statistic in ("kfoci", "chatterjee"):
        return forward_selection(X, Y, k_nn, "chatterjee")
    elif statistic == "MDCSIS":
        return mdc_select(X, Y, nsis=nsis)
    elif statistic == "BcorSIS":
        return bcorsis_select(X, Y, nsis=nsis)
    elif statistic == "Kfilter":
        return kfilter_select(X, Y, nsis=nsis)
    else:
        raise ValueError(f"Unknown statistic: {statistic!r}")


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------

def run_experiment(setting, true_set={0, 1, 2}, n_sim=200, n=200, p=10, k_nn=10, statistic="linear"):
    nsis = len(true_set)
    exact = 0
    contains = 0
    size_sum = 0

    for _ in tqdm(range(n_sim), desc=f"Running: setting={setting}, statistic={statistic}, p={p}, n={n}", leave=False):
        X, Y = data_generation(n, p, setting)
        selected = _select_features(X, Y, k_nn, statistic, nsis)
        selected_set = set(selected)

        if selected_set == true_set:
            exact += 1
        if true_set.issubset(selected_set):
            contains += 1
        size_sum += len(selected)

    return exact / n_sim, contains / n_sim, size_sum / n_sim


def single_sim(setting, true_set, n, p, k_nn, statistic, seed):
    np.random.seed(seed)
    nsis = len(true_set)
    X, Y = data_generation(n, p, setting)
    selected = _select_features(X, Y, k_nn, statistic, nsis)
    selected_set = set(selected)
    return int(selected_set == true_set), int(true_set.issubset(selected_set)), len(selected)


def run_experiment_parallel(setting, true_set={0, 1, 2}, n_sim=200, n=200, p=10, k_nn=10, statistic="linear", global_seed=42, n_jobs=-1):
    seed_sequence = np.random.SeedSequence(global_seed)
    child_seeds = seed_sequence.spawn(n_sim)
    seeds = [int(s.generate_state(1)[0]) for s in child_seeds]

    results = Parallel(n_jobs=n_jobs)(
        delayed(single_sim)(setting, true_set, n, p, k_nn, statistic, seed)
        for seed in tqdm(seeds, desc=f"Running: setting={setting}, statistic={statistic}, p={p}, n={n}", leave=True)
    )

    exact_total = sum(r[0] for r in results)
    contains_total = sum(r[1] for r in results)
    size_sum = sum(r[2] for r in results)

    return exact_total / n_sim, contains_total / n_sim, size_sum / n_sim


def run_all_experiments(settings_list, statistics_list, p_list, n_list, true_set={0, 1, 2}, n_sim=200, k_nn=10, parallel_indic=False, global_seed=42):
    results = []
    runner = run_experiment_parallel if parallel_indic else run_experiment

    for setting in settings_list:
        for p in p_list:
            for statistic in statistics_list:
                for n in n_list:
                    kwargs = dict(setting=setting, true_set=true_set, n_sim=n_sim, n=n, p=p, k_nn=k_nn, statistic=statistic)
                    if parallel_indic:
                        kwargs["global_seed"] = global_seed
                    exact, contains, avg_size = runner(**kwargs)
                    results.append({
                        "setting": setting, "statistic": statistic, "p": p, "n": n,
                        "exact_recovery_rate": exact, "containment_rate": contains, "avg_selected_size": avg_size,
                    })

    return pd.DataFrame(results)
