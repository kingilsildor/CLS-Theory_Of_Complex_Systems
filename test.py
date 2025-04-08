import matplotlib.pyplot as plt
import numpy as np

# Load data
data = np.loadtxt("data/Data_neuron.txt", dtype=float)
waiting_time = np.diff(data)

# Create histogram bins
num_bins = 50
hist, bin_edges = np.histogram(waiting_time, bins=num_bins)
x = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin centers
y = hist

# Remove zero values to avoid log issues
nonzero_indices = y > 0
x = x[nonzero_indices]
y = y[nonzero_indices]

# Transform y to ln(y)
log_y = np.log(y)

# Fit a first-degree polynomial to (x, ln(y))
p = np.polyfit(x, log_y, 1)

# Extract coefficients
a = np.exp(p[1])  # e^(intercept)
b = p[0]  # slope

# Create fitted curve
y_fit = a * np.exp(b * x)

# Plot histogram and fitted curve
plt.bar(
    x,
    y,
    width=(bin_edges[1] - bin_edges[0]),
    alpha=0.6,
    color="blue",
)
plt.plot(x, y_fit, label=f"Fit: ${a:.2f} e^{{{b:.2f} x}}$", color="red")
plt.xlabel("Waiting Time Bins")
plt.ylabel("Frequency")
plt.legend()
plt.title("Exponential Fit using numpy.polyfit")
plt.show()
