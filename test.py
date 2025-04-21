import matplotlib.pyplot as plt
import numpy as np

tau0 = 1.9  # Refractory period (ms)
lambda_ = 0.08  # Decay rate (ms^-1)
N = 1000  # Number of spikes to simulate

# Generate ISIs from delayed exponential distribution
u = np.random.rand(N)
simulated_isis = tau0 - np.log(u) / lambda_

# Load original spike times and compute ISIs
original_spikes = np.loadtxt("data/Data_neuron.txt")  # Replace with actual path
original_isis = np.diff(original_spikes)
original_isis = np.sort(original_isis)[::-1]

plt.figure(figsize=(10, 6))

# Plot original data ISIs
plt.hist(
    original_isis,
    bins=50,
    color="blue",
    density=True,
    label="Original Data",
)
# Plot simulated ISIs
plt.hist(
    simulated_isis,
    bins=50,
    alpha=0.8,
    color="red",
    density=True,
    label="Simulated Data",
)

plt.xlabel("Inter-Spike Interval (τ) [ms]")
plt.ylabel("Probability Density")
plt.title("Comparison of Original vs. Simulated ISIs")
plt.legend()
plt.show()
