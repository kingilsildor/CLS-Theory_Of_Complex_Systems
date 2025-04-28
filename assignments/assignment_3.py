import itertools

import matplotlib.pyplot as plt
import numpy as np
from config import FIG_DPI
from load_data import load_data
from matplotlib.lines import Line2D
from scipy.stats import binom

plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 12

s_data = load_data("US_SupremeCourt_n9_N895.txt", "str")
s_data[s_data == 0] = -1

N, n = s_data.shape
hi_data = load_data("hi_ussc_unsorted.txt", "float")
Jij_data = load_data("Jij_ussc_unsorted.txt", "float")

unique_spins = np.unique(s_data, axis=0)
pairs = list(itertools.combinations(range(n), 2))


def energy(s, h, J, pairs):
    """Calculate energy of spin configuration s."""
    if s.ndim == 1:
        s = s.reshape(1, -1)

    field_term = np.dot(s, h)

    interaction_term = np.zeros(s.shape[0])
    for k, (i, j) in enumerate(pairs):
        interaction_term += J[k] * s[:, i] * s[:, j]

    return field_term + interaction_term


def partition_function(h, J, pairs):
    """Calculate the partition function Z."""
    n = len(h)
    arr = np.array([-1, 1])
    results = np.array(np.meshgrid(*[arr] * n)).T.reshape(-1, n)
    Z = np.zeros(results.shape[0])
    Z = np.exp(energy(results, h, J, pairs))
    Z = np.sum(Z)
    return Z


def true_probability(s, h, J, pairs):
    """Calculate the probability p_g(s) of a configuration s."""
    Z = partition_function(h, J, pairs)
    energy_values = energy(s, h, J, pairs)
    return np.exp(energy_values) / Z


def empirical_probability(s_data, s, N):
    """Calculate the empirical probability p_D(s) of a configuration s."""
    matches = np.all(s_data[:, np.newaxis] == s, axis=2)
    return np.sum(matches, axis=0) / N


def calc_empirical_average_spin(s):
    """Calculate the average spin <s_i>_D."""
    return np.mean(s, axis=0)


def calc_true_average_spin(s_data, s, h, J, pairs):
    """Calculate the average spin <s_i>."""
    p_g = true_probability(s_data, h, J, pairs)
    prob = s * p_g[:, np.newaxis]
    return np.sum(prob, axis=0)


def calc_average_votebehavior(s_data, col=True, conservative=True):
    axis = 0 if col else 1
    vote_behavior = 1 if conservative else -1
    return np.sum(s_data == vote_behavior, axis=axis) / s_data.shape[axis]


def calc_emperic_average_spin_pair(s, N, flatten=False):
    """Calculate the average spin <s_i s_j>."""
    prob = np.dot(s.T, s) / N
    if flatten:
        return prob.flatten()
    return prob


def calc_true_average_spin_pair(s, h, J, pairs, flatten=False):
    """Calculate the average spin <s_i s_j>."""
    p_g = true_probability(s, h, J, pairs)
    n = s.shape[1]
    avg_spin_pair = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            avg_spin_pair[i, j] = np.sum(s[:, i] * s[:, j] * p_g)

    if flatten:
        return avg_spin_pair.flatten()
    return avg_spin_pair


def P_I(p, k):
    N = len(p)

    combinations = list(itertools.combinations(range(N), k))
    prob = np.zeros(len(combinations))

    for i, combination in enumerate(combinations):
        prob_conservative = [p[i] for i in combination]
        prob_liberal = [1 - p[i] for i in range(N) if i not in combination]

        prob[i] = np.prod(prob_conservative) * np.prod(prob_liberal)

    return np.sum(prob)


def P_I_binom(p, k):
    mean_p = np.mean(p)
    N = len(p)
    return binom.pmf(k, N, mean_p)


def P_D(s_data):
    num_conservative = (np.sum(s_data, axis=1) + 9) // 2
    _, counts = np.unique(num_conservative, return_counts=True)
    return counts / len(s_data)


def get_J_matrix(J, pairs, n):
    J_matrix = np.zeros((n, n))
    for k, (i, j) in enumerate(pairs):
        J_matrix[i, j] = J[k]
        J_matrix[j, i] = J[k]
    return J_matrix


def plot_averages(spin_vector, spin_matrix, save=True):
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], width_ratios=[10, 1])

    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.plot(spin_vector, marker="o", linestyle="None", color="black")
    ax_top.set_ylim(-0.50, 0.50)
    ax_top.set_ylabel(r"$\langle s_i \rangle$")
    ax_top.tick_params(axis="x", labelbottom=False)

    ax_bottom = fig.add_subplot(gs[1, 0])
    ax_bottom.xaxis.tick_top()
    im = ax_bottom.imshow(spin_matrix, cmap="gray_r", vmin=0, vmax=1)

    cax = fig.add_subplot(gs[1, 1])
    fig.colorbar(im, cax=cax, label=r"$\langle s_i s_j \rangle$")

    plt.tight_layout()
    if save:
        plt.savefig("results/average_spin_combined.png", dpi=FIG_DPI)
        plt.close()
    else:
        plt.show()


def plot_interactions(h_vector, J_matrix, save=True):
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], width_ratios=[10, 1])

    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.plot(h_vector, marker="o", linestyle="None", color="black")
    ax_top.set_ylim(-0.55, 0.55)
    ax_top.set_ylabel(r"$h_i$")
    ax_top.tick_params(axis="x", labelbottom=False)

    ax_bottom = fig.add_subplot(gs[1, 0])
    ax_bottom.xaxis.tick_top()
    im = ax_bottom.imshow(J_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    cax = fig.add_subplot(gs[1, 1])
    fig.colorbar(im, cax=cax, label=r"$J_{ij}$")

    plt.tight_layout()
    if save:
        plt.savefig("results/interactions_combined.png", dpi=FIG_DPI)
        plt.close()
    else:
        plt.show()


def plot_cross_validation(true, pred, rmse, fig_name, save=True):
    plt.figure(figsize=(8, 6))
    plt.scatter(true, pred, c="blue", label="States", alpha=0.6)
    # for i in range(len(true)):
    #     plt.text(true[i], pred[i], str(i), fontsize=10, color="black")

    plt.plot([0, 1], [0, 1], "r--", label="Perfect fit", alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel("Model Probability p_g(s)")
    plt.ylabel("Empirical Probability p_D(s)")
    plt.title(f"Cross-validation: Empirical vs Model Probabilities\nRMSE = {rmse:.4f}")
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig(f"results/{fig_name}.png", dpi=FIG_DPI)
        plt.close()
    else:
        plt.show()


def plot_vote_behavior(k_list, binom_approx, save=True):
    plt.figure(figsize=(8, 6))

    plt.plot(
        k_list, marker="o", linestyle="None", color="black", label=r"$P_I(k)$ data"
    )
    plt.plot(binom_approx, label="Binomial Approximation", color="red")
    plt.xlabel("Number of conservative votes")
    plt.ylabel(r"Probability $P_I(k)$")
    plt.title(r"Probability of $k$ conservative votes")
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig("results/voting_behavior.png", dpi=FIG_DPI)
        plt.close()
    else:
        plt.show()


def plot_vote_average(p, save=True):
    mean_p = np.mean(p)
    std_p = np.std(p)
    std_devs = [1, 2, 3]
    colors = ["orange", "green", "purple"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(p, np.zeros_like(p), color="blue", alpha=0.6, label="Data points")

    for i, n in enumerate(std_devs):
        ax.axvline(
            mean_p + n * std_p,
            color=colors[i],
            linestyle="--",
            linewidth=1.5,
        )
        ax.axvline(
            mean_p - n * std_p,
            color=colors[i],
            linestyle="--",
            linewidth=1.5,
        )

    ax.axvline(mean_p, color="red", linestyle="-", linewidth=2, label="Mean")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    custom_lines = [
        Line2D([0], [0], color="blue", lw=0, marker="o", label="Data points"),
        Line2D([0], [0], color="red", lw=2, label="Mean"),
        Line2D([0], [0], color="orange", linestyle="--", lw=1.5, label=r"$M+1\sigma$"),
        Line2D([0], [0], color="green", linestyle="--", lw=1.5, label=r"$M+2\sigma$"),
        Line2D([0], [0], color="purple", linestyle="--", lw=1.5, label=r"$M+3\sigma$"),
    ]
    ax.legend(handles=custom_lines)
    plt.tight_layout()

    if save:
        plt.savefig("results/voting_average.png", dpi=FIG_DPI)
        plt.close()
    else:
        plt.show()


def plot_vote_comparison(P_D_list, P_I_list, save=True):
    plt.figure(figsize=(8, 6))
    plt.plot(P_D_list, marker="o", label="$P_D(k)$ (Data)")
    plt.plot(P_I_list, marker="o", label="$P_I(k)$ (Independent)")
    plt.xlabel("$k$ (Conservative votes)")
    plt.ylabel("Probability")
    plt.title("Vote Comparison")
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig("results/vote_comparison.png", dpi=FIG_DPI)
        plt.close()
    else:
        plt.show()


# ############### 6.1
# print(f"Number of datapoints: {N}")
# print(f"Number of spins: {n}")
# print(f"Number of possible unique spins: {2**n}")
# print(f"Number of unique spins in the dataset: {unique_spins.shape[0]}")

# ############### 6.3
# average_spin = calc_empirical_average_spin(s_data)
# indices = np.argsort(average_spin)
# average_spin = average_spin[indices]
# average_spin_pair = calc_emperic_average_spin_pair(s_data[:, indices], N)
# plot_averages(average_spin, average_spin_pair)

# # ################ 6.4
# hi_data = hi_data[indices]
# Jij_matrix = get_J_matrix(Jij_data, pairs, n)
# Jij_matrix = Jij_matrix[indices][:, indices]
# plot_interactions(hi_data, Jij_matrix)

# # ############### 6.5
# single_spin = unique_spins[0]

# p_g = true_probability(unique_spins, hi_data, Jij_data, pairs)
# p_D = empirical_probability(s_data, unique_spins, N)

# rmse = np.sqrt(mean_squared_error(p_g, p_D))
# plot_cross_validation(p_g, p_D, rmse, "true_vs_empirical_probabilities")

# ############### 6.6
# ### A
# avg_spin = calc_true_average_spin(unique_spins, unique_spins, hi_data, Jij_data, pairs)
# avg_spin_D = calc_empirical_average_spin(s_data)
# rmse = np.sqrt(mean_squared_error(avg_spin, avg_spin_D))
# plot_cross_validation(
#     avg_spin[indices], avg_spin_D[indices], rmse, "true_vs_empirical_avg_spin"
# )
# ### B
# avg_spin_pair = calc_true_average_spin_pair(unique_spins, hi_data, Jij_data, pairs)
# avg_spin_pair_D = calc_emperic_average_spin_pair(s_data, N)
# rmse = np.sqrt(mean_squared_error(avg_spin_pair, avg_spin_pair_D))
# plot_cross_validation(
#     avg_spin_pair[indices].flatten(),
#     avg_spin_pair_D[indices].flatten(),
#     rmse,
#     "true_vs_empirical_avg_spin_pair",
# )

# ############### 6.7
# p = calc_average_votebehavior(s_data)
# plot_vote_average(p)

# P_I_list = np.array([P_I(p, k) for k in range(n + 1)])
# binom_approx = np.array([P_I_binom(p, k) for k in range(n + 1)])
# plot_vote_behavior(P_I_list, binom_approx)

# ############### 6.8

# P_D_list = P_D(s_data)
# plot_vote_comparison(P_D_list, P_I_list)
