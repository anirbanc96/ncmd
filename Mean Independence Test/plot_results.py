import pickle
from utils import plot_power_curves

# ── Colour / marker maps ──────────────────────────────────────────────────────
COLORS_UNI = {
    "MDD":            "red",
    "dCov":           "purple",
    "pMIT [r=0.8]":   "orange",
    "pMIT [formula]": "brown",
    "Chatterjee":     "green",
    "NCMD [k=5]":     "dodgerblue",
    "NCMD [k=10]":    "navy",
}
MARKERS_UNI = {
    "MDD":            "P",
    "dCov":           "s",
    "pMIT [r=0.8]":   "X",
    "pMIT [formula]": "H",
    "Chatterjee":     "p",
    "NCMD [k=5]":     "o",
    "NCMD [k=10]":    "*",
}

COLORS_MULTI = {
    "MDD":            "red",
    "dCov":           "purple",
    "pMIT [r=0.8]":   "orange",
    "pMIT [formula]": "brown",
    "AC":             "green",
    "NCMD [k=5]":     "dodgerblue",
    "NCMD [k=10]":    "navy",
}
MARKERS_MULTI = {
    "MDD":            "P",
    "dCov":           "s",
    "pMIT [r=0.8]":   "X",
    "pMIT [formula]": "H",
    "AC":             "p",
    "NCMD [k=5]":     "o",
    "NCMD [k=10]":    "*",
}

# ── Setting labels ────────────────────────────────────────────────────────────
SETTING_LABELS = {
    "Linear":          "Linear [Power]",
    "Step":            "Step [Power]",
    "W shaped":        "W-Shaped [Power]",
    "Sinusoid":        "Sinusoid [Power]",
    "Circular":        "Circular [Type-I]",
    "Heteroskedastic": "Heteroskedastic [Type-I]",
}

SETTING_LABELS_MULTI = {
    "Noise":                  "Noise [Type-I]",
    "Heteroskedastic":        "Heteroskedastic [Type-I]",
    "Nonlinear Additive":     "Nonlinear Additive [Power]",
    "Interaction":            "Interaction [Power]",
    "Interaction & Additive": "Interaction & Additive [Power]",
    "Radial":                 "Radial [Power]",
}

# ── Data files ────────────────────────────────────────────────────────────────
DATASETS = [
    # (pkl_path, labels, colors, markers, col_size, row_size, out_pdf)
    (
        "results_uniform_univariate_d1_n250_nsim200.pkl",
        SETTING_LABELS, COLORS_UNI, MARKERS_UNI,
        2.0, 1.5,
        "results/power_curves_uniform.pdf",
    ),
    (
        "results_normal_univariate_d1_n250_nsim200.pkl",
        SETTING_LABELS, COLORS_UNI, MARKERS_UNI,
        2.0, 1.5,
        "results/power_curves_normal.pdf",
    ),
    (
        "results_beta_univariate_d1_n250_nsim200.pkl",
        SETTING_LABELS, COLORS_UNI, MARKERS_UNI,
        2.0, 1.5,
        "results/power_curves_beta.pdf",
    ),
    (
        "results_uniform_multivariate_d10_n250_nsim200.pkl",
        SETTING_LABELS_MULTI, COLORS_MULTI, MARKERS_MULTI,
        2.0, 1.5,
        "results/power_curves_uniform_multivariate.pdf",
    ),
    (
        "results_normal_multivariate_d10_n250_nsim200.pkl",
        SETTING_LABELS_MULTI, COLORS_MULTI, MARKERS_MULTI,
        2.0, 1.5,
        "results/power_curves_normal_multivariate.pdf",
    ),
    (
        "results_beta_multivariate_d10_n250_nsim200.pkl",
        SETTING_LABELS_MULTI, COLORS_MULTI, MARKERS_MULTI,
        2.0, 1.5,
        "results/power_curves_beta_multivariate.pdf",
    ),
]

# ── Plot ──────────────────────────────────────────────────────────────────────
for pkl_path, labels, colors, markers, col_size, row_size, out_pdf in DATASETS:
    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)

    plot_power_curves(
        saved["results"],
        labels,
        colors,
        markers,
        saved["lambda_grid"],
        save_path=out_pdf,
        col_size=col_size,
        row_size=row_size,
    )
