from model1 import *
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

D1_DIM = 2
D2_DIM = 4
A_BOUND = 2

n_samples = 1000
K_neighbors = 5
theta_vals = np.linspace(0.5, 1.5, 8)
n_trials = 25

pop_vals = np.zeros((len(theta_vals), 3))
est_means = np.zeros((len(theta_vals), 3))
est_stds = np.zeros((len(theta_vals), 3))

for i, t in enumerate(theta_vals):
    e1, e2, e2_13 = compute_fully_modular_population(t, d1=D1_DIM, d2=D2_DIM, a=A_BOUND)
    pop_vals[i] = [e1, e2, e2_13]

    trials = []
    for _ in tqdm(range(n_trials)):
        X1, X2, X3, Y = generate_fully_modular_data(n_samples, t, d1=D1_DIM, d2=D2_DIM, a=A_BOUND)
        hat_e1, hat_e2, hat_e2_13 = estimate_all_sobol_indices(X1, X2, X3, Y, K_neighbors)
        trials.append([hat_e1, hat_e2, hat_e2_13])

    trials = np.array(trials)
    est_means[i] = np.mean(trials, axis=0)
    est_stds[i] = np.std(trials, axis=0)


plt.figure(figsize=(3.5, 3))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

labels = [r'$\eta_{X_1}$', r'$\eta_{X_2}$', r'$\eta_{2}$']

for idx in range(3):

    plt.plot(theta_vals, pop_vals[:, idx], '-', color=colors[idx], linewidth=2)

    plt.plot(theta_vals, est_means[:, idx], '--', marker='o', markersize=4, color=colors[idx], alpha=0.8)

    plt.fill_between(theta_vals,
                     est_means[:, idx] - est_stds[:, idx],
                     est_means[:, idx] + est_stds[:, idx],
                     color=colors[idx], alpha=0.15)


color_handles = [mlines.Line2D([], [], color=c, label=l) for c, l in zip(colors, labels)]
leg1 = plt.legend(handles=color_handles, loc='upper left', title="", labelspacing=0.001)
plt.gca().add_artist(leg1)


line_handles = [
    mlines.Line2D([], [], color='black', linestyle='-', label='Actual'),
    mlines.Line2D([], [], color='black', linestyle='--', marker='o', label='Estimate')
]
plt.legend(handles=line_handles, bbox_to_anchor = (0,0.55), loc='center left', title="", labelspacing = 0.01)

plt.title(f'Estimating Sobol Indices', fontsize=13)
plt.xlabel(r'Tuning Parameter $\theta$', fontsize=11)
plt.ylabel('Sobol Index', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("Model1.pdf", dpi=1200)
plt.show()