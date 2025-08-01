import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm
from scipy.stats import uniform


def get_3D_probs(x_grid, y_grid, z_grid, mu_x, sigma_x, mu_y, sigma_y, z_range):
    a, b = z_range

    # Compute CDF differences (probability mass over each grid cell)
    px = norm.cdf(x_grid[1:], loc=mu_x, scale=sigma_x) - norm.cdf(
        x_grid[:-1], loc=mu_x, scale=sigma_x
    )
    py = norm.cdf(y_grid[1:], loc=mu_y, scale=sigma_y) - norm.cdf(
        y_grid[:-1], loc=mu_y, scale=sigma_y
    )
    pz = uniform.cdf(z_grid[1:], loc=a, scale=b - a) - uniform.cdf(
        z_grid[:-1], loc=a, scale=b - a
    )

    # Form the full grid of probabilities
    probs = (
        np.outer(px, py)[:, :, None] * pz[None, None, :]
    )  # shape (len(px), len(py), len(pz))
    return probs


def simulate_multinomial_from_grid(n_samples, probs):
    # Normalize to ensure it's a valid multinomial distribution
    flat_probs = probs.ravel()
    flat_probs /= flat_probs.sum()

    # Sample from multinomial
    sample = np.random.multinomial(n_samples, flat_probs)

    return sample.reshape(probs.shape)


def estimate_median_from_histogram(bin_edges, counts):
    counts = np.asarray(counts)
    bin_edges = np.asarray(bin_edges)
    cumsum = np.cumsum(counts)
    total = cumsum[-1]
    half = total / 2

    # Find bin where median lies
    m = np.searchsorted(cumsum, half)

    # Handle edge case if median in first bin
    cumsum_prev = 0 if m == 0 else cumsum[m - 1]

    # Bin width
    w = bin_edges[m + 1] - bin_edges[m]

    # Linear interpolation within bin
    median = bin_edges[m] + w * (half - cumsum_prev) / counts[m]
    return median


def estimate_mean_from_histogram(bin_edges, counts):
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    return np.sum(bin_centers * counts / counts.sum())


def estimate_var_from_histogram(bin_edges, counts, _mean=None):
    if _mean is None:
        _mean = estimate_mean_from_histogram(bin_edges, counts)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    return np.sum((bin_centers - _mean) ** 2 * counts / counts.sum())


def estimate_stdev_from_histogram(*args, **kwargs):
    return np.sqrt(estimate_var_from_histogram(*args, **kwargs))


def get_marginals(counts):
    indices = set(range(len(counts.shape)))
    return [counts.sum(tuple(indices - set([i]))) for i in range(len(counts.shape))]


# Example
bin_edges = np.array([0, 1, 2, 3, 4, 5])
counts = np.array([5, 10, 20, 15, 0])
print("Robust median estimate:", median_from_histogram(bin_edges, counts))


x_grid = np.arange(-10, 11)
y_grid = np.arange(-10, 11)
z_grid = np.arange(-10, 11)

probs = get_3D_probs(x_grid, y_grid, z_grid, 0, 2, 0, 2, (-1, 2))
counts = simulate_multinomial_from_grid(100, probs)

x_marg, y_marg, z_marg = get_marginals(counts)

estimate_median_from_histogram(x_grid, x_marg)
estimate_mean_from_histogram(x_grid, x_marg)
estimate_stdev_from_histogram(x_grid, x_marg)
estimate_stdev_from_histogram(y_grid, y_marg)


plt.matshow(counts.sum(axis=2))
plt.show()
