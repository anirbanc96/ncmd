import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from joblib import Parallel, delayed
import time
import pickle
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

def generate_data(n, lam, setting, rng, X_setting = "uniform"):
    
    if X_setting == "uniform":
        X = rng.uniform(-1, 1, size=n)
    elif X_setting == "normal":
        X = rng.normal(0, 1, size = n)
    elif X_setting == "beta":
        alpha = 0.1  
        U = rng.beta(alpha, alpha, size=n)
        X = 2 * U - 1  


    if setting == "Linear":
        Y = rng.normal(0, 3 * lam, size=n) + 0.5 * X

    elif setting == "Step":
        step = np.where(X <= -0.5,  -3,
               np.where(X <=  0.0,   2,
               np.where(X <=  0.5,  -4, -3)))
        Y = step + 10 * lam * rng.standard_normal(n)

    elif setting == "W shaped":
        step = np.where(X < 0, np.abs(X + 0.5), np.abs(X - 0.5))
        Y = step + 0.75 * lam * rng.standard_normal(n)

    elif setting == "Sinusoid":
        Y = np.cos(8 * np.pi * X) + 3 * lam * rng.standard_normal(n)

    elif setting == "Circular":
        Z = 2 * rng.binomial(1, 0.5, size=n) - 1          
        Y = Z * np.sqrt(np.clip(1 - X**2, 0, None)) + 0.9 * lam * rng.standard_normal(n)

    elif setting == "Heteroskedastic":
        step = np.where(np.abs(X) <= 0.5, 1, 0)
        Y = 3 * (step * (2 - lam) + lam) * rng.standard_normal(n)

    else:
        raise ValueError(f"Unknown setting: '{setting}'.")

    return X.reshape(n, 1), Y.reshape(n, 1)

def generate_data_multivariate(n, d, lam, setting, rng, X_setting="uniform"):
    
    if X_setting == "uniform":
        X = rng.uniform(-1, 1, size=(n, d))
    elif X_setting == "normal":
        X = rng.normal(0, 1, size=(n, d))
    elif X_setting == "beta":
        alpha = 0.1  
        U = rng.beta(alpha, alpha, size=(n, d))
        X = 2*U - 1
    else:
        raise ValueError(f"Unknown X_setting '{X_setting}'")

    if setting == "Noise":
        Y = rng.normal(0, lam, size=n)  # mean independent

    elif setting == "Heteroskedastic":

        Z   = 2 * rng.binomial(1, 0.5, size=n) - 1                    
        f_X = 1 + 2 * np.sum(np.abs(X ** 2), axis=1)                      
        Y   = Z * f_X + lam * rng.standard_normal(n)

    elif setting == "Nonlinear Additive":

        Y = np.sin(np.pi*X[:, 0]) + np.log(np.abs(X[:, 1])+1) + lam * rng.standard_normal(n)

    elif setting == "Interaction":

        Y = X[:, 0] * X[:, 1] + lam * rng.standard_normal(n)

    elif setting == "Radial":

        d = X.shape[1]
        if d < 5:
            raise ValueError("Radial setting requires d >= 5")

        idx = rng.choice(d, size=5, replace=False)   # random 5 indices
        r = np.linalg.norm(X[:, idx], axis=1)/np.sqrt(5)

        Y = np.cos(r) + lam * rng.standard_normal(n)

    elif setting == "Interaction & Additive":
        Y = np.sin(X[:,0]) + np.cos(X[:,1]) * X[:, 2] + lam * rng.standard_normal(n)

    else:
        raise ValueError(f"Unknown setting '{setting}'. Choose from "
                         "'Noise', 'Heteroskedastic', 'Nonlinear Additive', "
                         "'Interaction', 'Radial'.")

    return X, Y.reshape(n, 1)

def _run_setting(setting, lambda_grid, methods, n, n_sim, X_setting, seed, d=1, dim_X="univariate"):

    rng          = np.random.default_rng(seed)
    method_names = list(methods.keys())
    setting_results = {
        m: {"mean": np.zeros(len(lambda_grid)), "sd": np.zeros(len(lambda_grid))}
        for m in method_names
    }

    timing = {m: {"total_time": 0.0, "count": 0} for m in method_names}

    for l_idx, lam in enumerate(lambda_grid):
        rejects = {m: [] for m in method_names}
        for _ in range(n_sim):
            if dim_X == "univariate":
                X, Y = generate_data(n, lam, setting, rng, X_setting)
            else:
                X, Y = generate_data_multivariate(n, d, lam, setting, rng, X_setting)
            for name, fn in methods.items():
                if name == "pMIT":
                    if dim_X == "univariate":
                        X, Y = generate_data(int(n), lam, setting, rng, X_setting)
                    else:
                        X, Y = generate_data_multivariate(int(n), d, lam, setting, rng, X_setting)
                    t0 = time.perf_counter()
                    result = fn(X, Y)
                    timing[name]["total_time"] += time.perf_counter() - t0
                    timing[name]["count"]      += 1
                    rejects[name].append(result["reject"])
                else:
                    t0 = time.perf_counter()
                    result = fn(X, Y)
                    timing[name]["total_time"] += time.perf_counter() - t0
                    timing[name]["count"]      += 1
                    rejects[name].append(result["reject"])

        for name in method_names:
            arr = np.array(rejects[name])
            setting_results[name]["mean"][l_idx] = np.mean(arr)
            setting_results[name]["sd"][l_idx]   = np.std(arr, ddof=1) / np.sqrt(n_sim)

        print(f"  [{setting}] lambda={lam:.2f}  " +
              "  ".join(f"{m}={setting_results[m]['mean'][l_idx]:.3f}±{setting_results[m]['sd'][l_idx]:.3f}"
                        for m in method_names),
              flush=True)

    avg_times = {
        m: (timing[m]["total_time"] / timing[m]["count"]) * 1000
        for m in method_names
    }

    print(f"  [{setting}] avg time/call (ms): " +
          "  ".join(f"{m}={avg_times[m]:.3f}" for m in method_names),
          flush=True)
    print(f"✓ {setting} done", flush=True)
    return setting, setting_results, avg_times

def run_experiment(lambda_grid, methods, SETTINGS, n=150, n_sim=500, seed=42, X_setting="uniform", dim_X="univariate", d_X=1, n_jobs=-1):
    lambda_grid  = np.asarray(lambda_grid)
    method_names = list(methods.keys())

    child_seeds = np.random.SeedSequence(seed).spawn(len(SETTINGS))
    seeds       = [int(s.generate_state(1)[0]) for s in child_seeds]

    print(f"Running {len(SETTINGS)} settings in parallel (n_jobs={n_jobs}) …\n")

    outcomes = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_run_setting)(setting, lambda_grid, methods, n, n_sim, X_setting, seeds[i], d=d_X, dim_X=dim_X)
        for i, setting in enumerate(SETTINGS)
    )

    results = {
        s: {m: {"mean": np.zeros(len(lambda_grid)), "sd": np.zeros(len(lambda_grid))}
            for m in method_names}
        for s in SETTINGS
    }

    timings = {s: {m: 0.0 for m in method_names} for s in SETTINGS}

    for setting, setting_results, avg_times in outcomes:
        results[setting] = setting_results
        timings[setting] = avg_times

    print("\n── Timing summary (avg ms / call) ──")
    header = f"{'Setting':<20}" + "".join(f"{m:>18}" for m in method_names)
    print(header)
    print("─" * len(header))
    for s in SETTINGS:
        row = f"{s:<20}" + "".join(f"{timings[s][m]:>17.3f}" for m in method_names)
        print(row)
    print()

    save_path = f"results_{X_setting}_{dim_X}_d{d_X}_n{n}_nsim{n_sim}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({"results": results, "timings": timings, "lambda_grid": lambda_grid}, f)
    print(f"Results saved → {save_path}")

    return results, timings

def plot_power_curves(results, SETTING_LABELS, COLORS, MARKERS, lambda_grid, save_path, col_size=2, row_size=2, alpha=0.05):

    lambda_grid  = np.asarray(lambda_grid)
    settings     = list(results.keys())
    method_names = list(next(iter(results.values())).keys())

    type1_settings = [s for s in settings if "[Type-I]" in SETTING_LABELS.get(s, "")]
    power_settings = [s for s in settings if "[Power]"  in SETTING_LABELS.get(s, "")]

    _ws = 0.4   
    fig_width  = (3 + _ws) * col_size + 1   
    fig_height = 2 * row_size + 1

    fig = plt.figure(figsize=(fig_width, fig_height))
    subfigs = fig.subfigures(1, 2, width_ratios=[1, 2 + _ws], wspace=0.2)

    subfigs[0].suptitle("Type I", fontsize=11, fontweight="bold")
    subfigs[1].suptitle("Power",  fontsize=11, fontweight="bold")

    axes_left  = subfigs[0].subplots(2, 1, gridspec_kw={"hspace": 0.45})
    axes_right = subfigs[1].subplots(2, 2, gridspec_kw={"hspace": 0.45, "wspace": _ws})

    def _fill_ax(ax, setting):
        for method in method_names:
            mean  = results[setting][method]["mean"]
            sd    = results[setting][method]["sd"]
            color = COLORS.get(method, "black")
            ax.plot(lambda_grid, mean, color=color, marker=MARKERS.get(method, "o"),
                    markersize=5, linewidth=1.8, label=method)
            ax.fill_between(lambda_grid,
                            np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                            color=color, alpha=0.15)
        ax.axhline(alpha, color="black", lw=1.2, ls="--")
        ax.set_xlim(lambda_grid[0] - 0.02, lambda_grid[-1] + 0.02)
        ax.set_ylim(-0.02, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
        ax.tick_params(labelsize=8)
        ax.set_xlabel(r"$\lambda$", fontsize=9)
        ax.set_ylabel("Rejection rate", fontsize=9)
        ax.yaxis.grid(True, color="#DDDDDD", lw=0.6)
        ax.xaxis.grid(True, color="#DDDDDD", lw=0.6)
        ax.set_axisbelow(True)
        label = SETTING_LABELS[setting].replace(" [Type-I]", "").replace(" [Power]", "")
        ax.set_title(label, fontsize=9, fontweight="bold", pad=5)

    for ax, setting in zip(axes_left, type1_settings):
        _fill_ax(ax, setting)

    for ax, setting in zip(axes_right.flatten(), power_settings):
        _fill_ax(ax, setting)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=COLORS.get(m, "black"), marker=MARKERS.get(m, "o"),
               markersize=6, linewidth=1.8, label=m)
        for m in method_names
    ]
    handles.append(Line2D([0], [0], color="black", lw=1.4, ls="--", label=rf"$\alpha$ = {alpha}"))
    n_per_row = int(np.ceil(len(handles) / 2))
    fig.legend(handles=handles, loc="lower center", ncol=n_per_row,
               fontsize=9.5, frameon=True, bbox_to_anchor=(0.5, -0.12))

    plt.savefig(save_path, dpi=1200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Plot saved → {save_path}")

def plot_times_table(times, save_path="method_times.pdf",col_width=1.2, row_height=0.4, fontsize=9,max_col_width=10, max_row_width=12):

    import textwrap

    settings     = list(times.keys())
    method_names = list(next(iter(times.values())).keys())

    def wrap(text, width):
        return "\n".join(textwrap.wrap(text, width))

    wrapped_settings = [wrap(s, max_col_width) for s in settings]
    wrapped_methods  = [wrap(m, max_row_width)  for m in method_names]

    cell_text = [
        [f"{times[s][m]:.3f}" for s in settings]
        for m in method_names
    ]

    n_cols = len(settings)
    n_rows = len(method_names)

    fig_w = col_width * n_cols + 1.8
    fig_h = row_height * (n_rows + 1)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText  = cell_text,
        rowLabels = wrapped_methods,
        colLabels = wrapped_settings,
        cellLoc   = "center",
        rowLoc    = "center",
        loc       = "center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)

    tbl.scale(1, (fig_h / (n_rows + 1)) / 0.22)

    ax.set_title("Average Execution Time (ms)",
                 fontsize=fontsize + 2, fontweight="bold",
                 pad=6, loc="center", x = 0.35, y=1.02)   

    with PdfPages(save_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Timing table saved → {save_path}")