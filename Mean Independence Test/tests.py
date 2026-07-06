import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
from scipy.stats import rankdata
from scipy.sparse import csr_matrix
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import chi2

def nncmi(X, Y, k, alpha=0.05):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    n = X.shape[0]

    if Y.shape[0] != n:
        raise ValueError("X and Y must have the same number of observations.")

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nbrs.fit(X)
    neighbors = nbrs.kneighbors(return_distance=False)[:, 1:]

    rows = np.repeat(np.arange(n), k)
    cols = neighbors.ravel()

    I = csr_matrix((np.ones(len(rows), dtype=np.uint8),
                    (rows, cols)),
                   shape=(n, n))

    Yi = Y[:, None, :]            
    Yj = Y[neighbors]             

    knn_sums = np.sum(Yi * Yj, axis=(1, 2))
    knn_term = np.mean(knn_sums / k)

    total = Y.sum(axis=0)
    total_sq = np.sum(Y * Y)

    global_term = (np.dot(total, total) - total_sq) / (n * (n - 1))

    Tn = knn_term - global_term

    mu = Y.mean(axis=0)
    Yc = Y - mu

    Yci = Yc[:, None, :]
    Ycj = Yc[neighbors]

    dots = np.sum(Yci * Ycj, axis=2)
    sigma1 = np.sum(dots**2)

    J = I.multiply(I.T).tocoo()

    if J.nnz > 0:
        mutual = np.sum(
            Yc[J.row] * Yc[J.col],
            axis=1
        )
        sigma1 += np.sum(mutual**2)

    sigma1 /= (n * k**2)

    indegree = np.bincount(cols, minlength=n)

    coef = (indegree / k - 1.0) ** 2

    proj = Yc @ mu

    sigma2 = np.mean(coef * proj**2)

    sigma_hat = sigma1 + sigma2

    if sigma_hat <= 0:
        raise ValueError("Estimated variance is non-positive.")

    Zn = np.sqrt(n) * Tn / np.sqrt(sigma_hat)

    return {
        "Zn": Zn,
        "reject": bool(Zn > norm.ppf(1 - alpha)),
    }


def chatterjee_test(X, Y, alpha = 0.05):

    X = X.ravel()
    Y = Y.ravel()

    order = np.argsort(X)
    Y_sorted = Y[order]

    ranks = rankdata(Y_sorted, method="ordinal")
    n = len(Y)

    diff = np.abs(np.diff(ranks))
    xi = 1 - (3 * np.sum(diff)) / (n**2 - 1)

    Zn = np.sqrt(n) * xi/np.sqrt(2/5)
    critical_value = norm.ppf(1 - alpha/2)
    reject = np.abs(Zn) > critical_value
    return {
        "Zn": Zn,
        "reject": reject
    }

def azadkia_chatterjee_test(X, Y, n_permutations=499, random_state=None):
    
    rng = np.random.default_rng(random_state)

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n = len(Y)
    if X.shape[0] != n:
        raise ValueError(f"X and Y must have same number of rows, got {X.shape[0]} and {n}.")

    _, idx = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)
    nn_idx = idx[:, 1]   

    def compute_T(Y_):

        R = rankdata(Y_, method="average")

        L = rankdata(-Y_, method="average")

        numerator   = np.sum(n * np.minimum(R, R[nn_idx]) - L**2)
        denominator = np.sum(L * (n - L))
        return numerator / denominator

    observed   = compute_T(Y)
    perm_stats = np.array([compute_T(Y[rng.permutation(n)]) for _ in range(n_permutations)])
    p_value    = (np.sum(perm_stats >= observed) + 1) / (n_permutations + 1)

    return {
        "T_n":    observed,
        "reject":  bool(p_value < 0.05),
    }


def mdd_test(X: np.ndarray, Y: np.ndarray, n_bootstrap: int = 499, random_state: int = None, alpha: float = 0.05):
    
    rng = np.random.default_rng(random_state)

    Y = np.asarray(Y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n = len(Y)
    if X.shape[0] != n:
        raise ValueError(f"X and Y must have the same number of observations, "
                         f"got {X.shape[0]} and {n}.")
    if n < 4:
        raise ValueError(f"n must be at least 4 for FMDD (denominator n(n-3)), got {n}.")

    def u_center(m):
        row_sum   = m.sum(axis=1)
        grand_sum = m.sum()

        row_mean   = row_sum / (n - 2)
        grand_mean = grand_sum / ((n - 1) * (n - 2))
        M = (m
             - row_mean[:, np.newaxis]
             - row_mean[np.newaxis, :]
             + grand_mean)
        np.fill_diagonal(M, 0.0)
        return M

    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]      
    a    = np.sqrt((diff ** 2).sum(axis=-1))
    A    = u_center(a)                                     


    diff_Y = Y[:, np.newaxis] - Y[np.newaxis, :]           
    b      = diff_Y ** 2 / 2                               
    B      = u_center(b)                                   

    AB = A * B

    fmdd_obs = AB.sum() / (n * (n - 3))
    T_obs    = n * fmdd_obs

    boot_stats = np.empty(n_bootstrap)
    for b_idx in range(n_bootstrap):
        eta             = rng.standard_normal(n)           
        boot_stats[b_idx] = (eta @ AB @ eta) / (n - 3)

    Q = np.quantile(boot_stats, 1 - alpha)
    reject  = bool(T_obs > Q)

    return {
        "mdd_stat": fmdd_obs,
        "reject":    reject,
    }


def dcov_test(X: np.ndarray, Y: np.ndarray, n_permutations: int = 499, random_state: int = None, alpha = 0.05):
    
    rng = np.random.default_rng(random_state)

    # --- Input validation and shaping ---
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError(f"X and Y must have the same number of observations, got {X.shape[0]} and {Y.shape[0]}.")

    def double_center(M: np.ndarray) -> np.ndarray:
        """Double-center a pairwise distance matrix."""
        row_mean   = M.mean(axis=1, keepdims=True)   
        col_mean   = M.mean(axis=0, keepdims=True)   
        grand_mean = M.mean()                         
        return M - row_mean - col_mean + grand_mean

    def pairwise_l2(Z: np.ndarray) -> np.ndarray:
        """Compute (n, n) matrix of pairwise Euclidean distances."""
        diff = Z[:, np.newaxis, :] - Z[np.newaxis, :, :]   
        return np.sqrt((diff ** 2).sum(axis=-1))

    A = double_center(pairwise_l2(X))

    def compute_dcov2(Y_: np.ndarray) -> float:
        """Compute dCov^2(X, Y_) for a given (possibly permuted) Y."""
        B = double_center(pairwise_l2(Y_))
        return n * (A * B).mean()

    observed_stat = compute_dcov2(Y)

    perm_stats = np.array([compute_dcov2(Y[rng.permutation(n)]) for _ in range(n_permutations)])

    p_value = (np.sum(perm_stats >= observed_stat) + 1) / (n_permutations + 1)

    return {
        "dcov2_stat": observed_stat,
        "reject":  p_value < alpha,
    }


def _solve_split_formula(N, N0=None):
    
    from scipy.optimize import brentq

    if N0 is None:
        N0 = 0.1 * N

    c = N0 / (2 * np.log(N0 / 2))   

    f = lambda x: x + c * np.log(x) - N

    x0 = brentq(f, 1.0, float(N) - 1e-6)
    n  = int(np.ceil(x0))
    m  = N - n

    return n, m

def pmit_test(X, Y, alpha=0.05, split_ratio=0.5, random_state=None, xgb_params=None):
    
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    N = len(Y)
    if X.shape[0] != N:
        raise ValueError(f"X and Y must have the same number of rows, "
                         f"got {X.shape[0]} and {N}.")

    if split_ratio == "formula":
        n_train, n_test = _solve_split_formula(N, N0=0.1 * N)
        actual_ratio    = n_train / N
    else:
        actual_ratio = split_ratio
        n_train      = int(np.round(split_ratio * N))
        n_test       = N - n_train

    X_D1, X_D2, Y_D1, Y_D2 = train_test_split(
        X, Y, test_size=n_test, random_state=random_state
    )
    n = len(Y_D2)

    default_xgb_params = {
        "n_estimators":     200,
        "max_depth":        6,
        "learning_rate":    0.1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "verbosity":        0,
        "random_state":     random_state,
    }
    if xgb_params is not None:
        default_xgb_params.update(xgb_params)

    model = XGBRegressor(**default_xgb_params)
    model.fit(X_D1, Y_D1)

    m_hat    = model.predict(X_D2)
    Y_bar    = Y_D2.mean()
    T_1n     = np.mean((m_hat - Y_bar) ** 2)
    sigma2_Y = Y.var()
    chi2_stat = n * T_1n / sigma2_Y

    p_value = float(1 - chi2.cdf(chi2_stat, df=1))
    reject  = bool(p_value < alpha)

    return {
        "chi2_stat":   float(chi2_stat),
        "p_value":     p_value,
        "reject":      reject,
        "split_ratio": actual_ratio,
        "n_train":     len(Y_D1),
        "n_test":      n,
    }