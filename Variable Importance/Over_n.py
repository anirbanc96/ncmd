import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

np.random.seed(42)

# ── DGP (original) ───────────────────────────────────────────────────────────

def generate_fully_modular_data(n, theta, d1, d2, a, sigma=0.2):
    X1 = np.random.uniform(-a, a, (n, d1))
    X2 = np.random.uniform(-a, a, (n, d2))
    X3 = np.random.uniform(-a, a, (n, 1))
    epsilon = np.random.normal(0, sigma, (n, 1))

    sum_X1 = np.sum(X1, axis=1, keepdims=True)
    sum_X2 = np.sum(X2, axis=1, keepdims=True)

    interaction_2way = sum_X1 * X3
    interaction_3way = sum_X1 * sum_X2 * X3

    Y = (theta * sum_X1
         + theta * sum_X2
         + theta * X3
         + (2 - theta) * interaction_2way
         + (2 - theta) * interaction_3way
         + epsilon)
    return X1, X2, X3, Y.flatten()


# ── Population truth ─────────────────────────────────────────────────────────

def compute_fully_modular_population(theta, d1, d2, a, sigma=0.2):
    v_X = (a**2) / 3.0

    v_f1   = (theta**2) * (d1 * v_X)
    v_f2   = (theta**2) * (d2 * v_X)
    v_f3   = (theta**2) * v_X
    v_f13  = ((2 - theta)**2) * (d1 * v_X) * v_X
    v_f123 = ((2 - theta)**2) * (d1 * v_X) * (d2 * v_X) * v_X
    v_eps  = sigma**2

    v_Y = v_f1 + v_f2 + v_f3 + v_f13 + v_f123 + v_eps

    eta1    = v_f1  / v_Y
    eta2    = v_f2  / v_Y
    eta2_13 = v_f13 / v_Y
    return eta1, eta2, eta2_13


# ── KNN estimators ───────────────────────────────────────────────────────────

def nn_expectation_term(X, Y, K=5):
    n = len(Y)
    nbrs = NearestNeighbors(n_neighbors=K + 1, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(X)
    nn_indices = indices[:, 1:]
    return np.mean(Y * np.sum(Y[nn_indices], axis=1)) / K


def estimate_all_sobol_indices(X1, X2, X3, Y, K=15):
    n = len(Y)
    denominator = np.mean(Y**2) - ((np.sum(Y)**2 - np.sum(Y**2)) / (n * (n - 1)))
    u_v_term    = (np.sum(Y)**2 - np.sum(Y**2)) / (n * (n - 1))

    def get_eta(X):
        return (nn_expectation_term(X, Y, K) - u_v_term) / denominator

    hat_eta1 = get_eta(X1)
    hat_eta2 = get_eta(X2)
    hat_eta3 = get_eta(X3)

    X13 = np.hstack((X1, X3))
    hat_eta_joint_13 = get_eta(X13)
    hat_eta2_13 = hat_eta_joint_13 - hat_eta1 - hat_eta3

    return hat_eta1, hat_eta2, hat_eta2_13


# ── Convergence experiment ────────────────────────────────────────────────────

def run_convergence_experiment(n_values, theta, d1, d2, a, sigma, K, n_reps):
    pop = compute_fully_modular_population(theta, d1, d2, a, sigma)
    records = {n: [] for n in n_values}

    for n in tqdm(n_values, desc='sample size'):
        for _ in range(n_reps):
            X1, X2, X3, Y = generate_fully_modular_data(n, theta, d1, d2, a, sigma)
            records[n].append(estimate_all_sobol_indices(X1, X2, X3, Y, K=K))

    means = np.array([[np.mean([r[i] for r in records[n]]) for n in n_values]
                      for i in range(3)])
    stds  = np.array([[np.std( [r[i] for r in records[n]]) for n in n_values]
                      for i in range(3)])
    return pop, means, stds


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_convergence(n_values, pop, means, stds, theta):
    labels  = [r'${\eta}_{X_1}$',
               r'${\eta}_{X_2}$',
               r'${\eta}_{2}$']
    colors  = ['#2563EB', '#D97706', '#059669']
    truths  = list(pop)

    fig, axes = plt.subplots(3, 1, figsize=(3, 4), sharex=True)
    fig.subplots_adjust(hspace=0.2)

    for ax, lbl, col, truth, mu, sd in zip(axes, labels, colors, truths, means, stds):
        ax.axhline(truth, color='black', lw=1.5, ls='--', zorder=3,
                   label=f'Actual')
        ax.plot(n_values, mu, color=col, lw=2, marker='o', ms=5,
                label='Estimate', zorder=4)
        ax.fill_between(n_values, mu - sd, mu + sd,
                        alpha=0.20, color = col)
        ax.set_xscale('log')
        # ax.set_ylabel('Sobol index', fontsize=11)
        ax.set_ylim(np.min(mu-sd), np.max(mu+sd))
        ax.set_title(lbl, fontsize=12, fontweight='bold', pad=6)
        ax.legend(fontsize=7, loc='lower right', bbox_to_anchor = (1.00, 1.00), labelspacing = 0.001)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)

    axes[-1].set_xlabel('$n$', fontsize=11)

    # fig.suptitle(
    #     r'KNN Sobol convergence  —  original DGP  ($\theta = %.1f$)' % theta,
    #     fontsize=13, y=1.01
    # )

    plt.savefig('sobol_convergence_original.pdf', dpi=1200,
                bbox_inches='tight', facecolor='white')
    plt.show()
    print("Figure saved → sobol_convergence_original.png")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    theta    = 1
    d1, d2   = 2, 4
    a        = 2.0
    sigma    = 0.2
    K        = 5
    n_reps   = 25
    n_values = [250, 500, 1_000, 2_000, 5_000, 10_000]

    pop, means, stds = run_convergence_experiment(
        n_values, theta, d1, d2, a, sigma, K, n_reps)

    print("\nPopulation Sobol indices (theta = %.1f):" % theta)
    print(f"  η1  [main X1]          = {pop[0]:.4f}")
    print(f"  η2  [main X2]          = {pop[1]:.4f}")
    print(f"  η13 [X1 × X3 interact] = {pop[2]:.4f}")

    plot_convergence(n_values, pop, means, stds, theta)