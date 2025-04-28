import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

from assignments.load_data import load_data

data = load_data("US_SupremeCourt_n9_N895.txt", "str")
data[data == 0] = -1  # Convert 0 (liberal) to -1

# Compute P_D(k)
conservative_votes = (np.sum(data, axis=1) + 9) // 2
k_values, counts = np.unique(conservative_votes, return_counts=True)
P_D = counts / len(data)

# Compute P_I(k) (Binomial approximation)
p_i = (np.mean(data, axis=0) + 1) / 2
mean_p = np.mean(p_i)
P_I = [binom.pmf(k, 9, mean_p) for k in range(10)]

# Plot
plt.bar(k_values - 0.2, P_D, width=0.4, label="$P_D(k)$ (Data)")
plt.bar(k_values + 0.2, P_I[: len(k_values)], width=0.4, label="$P_I(k)$ (Independent)")
plt.xlabel("$k$ (Conservative votes)")
plt.ylabel("Probability")
plt.legend()
plt.show()
