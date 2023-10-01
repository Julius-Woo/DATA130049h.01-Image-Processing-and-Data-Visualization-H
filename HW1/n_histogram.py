import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def evaluate_histogram(data_np, no_bins, data_min, data_max):
    """
    Evaluate an n-dimensional histogram.
    
    Parameters:
        - data_np: np.ndarray of shape (dimension, data)
        - no_bins: list of bins count for each dimension
        - data_min: list of minimum for each dimension
        - data_max: list of maximum for each dimension

    Returns:
        - histogram: np.ndarray of shape specified by no_bins
    """

    dimension = data_np.shape[0]
    numdata = data_np.shape[1] # Number of data points
    
    # Initialize histogram, an all-zero n-D array of size no_bins
    histogram = np.zeros(tuple(no_bins), dtype=int)
    
    # Compute bin spacings for each dimension
    bin_spacings = [(data_max[i] - data_min[i]) / no_bins[i] for i in range(dimension)]
    
    # Compute bin position for each point in data and increment histogram count
    for i_data in range(numdata):  # Loop over data points
        bin_pos = []  # Bin position for this data point
        for i_dim in range(dimension):  # Loop over dimensions
            value = data_np[i_dim][i_data]
            b = int((value - data_min[i_dim]) / bin_spacings[i_dim])  # Bin index
            # Prevent out-of-bound indices
            b = max(b, 0)
            b = min(b, no_bins[i_dim] - 1)
            bin_pos.append(b)
        
        # Increment histogram count at the computed position
        histogram[tuple(bin_pos)] += 1
    
    return histogram


def bins_power_of_two(n):
    
    """
    Determine number of bins as a power of 2.
    """
    k = np.log2(np.log2(n))
    
    return int(2**np.ceil(k))


# Generate a 2D Gaussian distribution with N data points
mean = [3, 5]
cov = [[1, 0], [0, 1]]  # diagonal covariance
N = 5000
x, y = np.random.multivariate_normal(mean, cov, N).T

data_np = np.array([x, y])

# call evaluate_histogram
m = bins_power_of_two(N) # Number of bins
no_bins = [m, m]
data_min = [-1, 1]
data_max = [7, 9]  # assume data are in range [-4, 4]

histogram = evaluate_histogram(data_np, no_bins, data_min, data_max)

histogram_normalized = histogram / N

# plot histogram

# 2D figure
plt.figure()
# calculate bin edges
x_bin_edges = np.linspace(data_min[0], data_max[0], no_bins[0]+1)
y_bin_edges = np.linspace(data_min[1], data_max[1], no_bins[1]+1)

# use imshow to plot 2D histogram
plt.imshow(histogram_normalized, cmap='viridis', origin='lower', 
        extent=[x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1]])
plt.colorbar(label="Normalized Frequency")
plt.title("Histogram for binormal distribution")

# Set x and y ticks to show bin edges
plt.xticks(x_bin_edges, rotation=90)
plt.yticks(y_bin_edges)

plt.xlabel("X value")
plt.ylabel("Y value")
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
plt.show()


# 3D figure
# Create the x, y coordinates for each bin center
x_positions = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2
y_positions = (y_bin_edges[:-1] + y_bin_edges[1:]) / 2

# Calculate bin widths and heights
x_widths = x_bin_edges[1] - x_bin_edges[0]
y_widths = y_bin_edges[1] - y_bin_edges[0]

# Create the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# For each bin, plot a bar
for i, x in enumerate(x_positions):
    for j, y in enumerate(y_positions):
        ax.bar3d(x - x_widths/2, y - y_widths/2, 0, x_widths, y_widths, 
                histogram_normalized[i, j], shade=True, color='CornflowerBlue')

# Set labels and title
ax.set_xlabel('X value')
ax.set_ylabel('Y value')
ax.set_zlabel('Frequency')
ax.set_title('3D Histogram for binormal distribution')

# Set x and y ticks to show bin edges
ax.set_xticks(x_bin_edges)
ax.set_yticks(y_bin_edges)

plt.show()