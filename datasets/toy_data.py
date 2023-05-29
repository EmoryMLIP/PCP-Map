import torch
import numpy as np
import scipy
import scipy.stats
import copy
from sklearn import datasets


"""
Functions for generating toy datasets as 
samples from target distributions.
"""


# generate half moons dataset
def get_moon(size):
    data = datasets.make_moons(n_samples=size, noise=0.1)[0]
    data = data.astype("float32")
    data = data * 2 + np.array([-1, -0.2])
    return torch.tensor(data, dtype=torch.float32)


# generate circles dataset
def get_circles(size):
    data = datasets.make_circles(n_samples=size, factor=.5, noise=0.08)[0]
    data = data.astype("float32")
    data *= 3
    return torch.tensor(data, dtype=torch.float32)


# generate swiss roll dataset
def get_swiss_roll(size, noise=1.0):
    data = datasets.make_swiss_roll(n_samples=size, noise=noise)[0]
    data = data.astype("float32")[:, [0, 2]]
    data /= 5
    return torch.tensor(data, dtype=torch.float32)


# generate 2 spirals dataset
def get_2spirals(size):
    n = np.sqrt(np.random.rand(size // 2, 1)) * 540 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(size // 2, 1) * 0.5
    d1y = np.sin(n) * n + np.random.rand(size // 2, 1) * 0.5
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
    x += np.random.randn(*x.shape) * 0.1
    return torch.tensor(x, dtype=torch.float32)


# generate checkerboard dataset
def get_checkerboard(size):
    x1 = np.random.rand(size) * 4 - 2
    x2_ = np.random.rand(size) - np.random.randint(0, 2, size) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    return torch.tensor(np.concatenate([x1[:, None], x2[:, None]], 1) * 2, dtype=torch.float32)


# generate spiral dataset
def get_spiral(size):
    # First draw some rotation samples from a beta distribution, then scale
    # them to the range between -pi and +2pi
    seeds = scipy.stats.beta.rvs(
        a=7,
        b=3,
        size=size) * 2 * np.pi - np.pi

    # Create a local copy of the rotations
    seeds_orig = copy.copy(seeds)

    # Re-normalize the rotations, then scale them to the range between [-3,+3]
    vals = (seeds + np.pi) / (3 * np.pi) * 6 - 3

    # Plot the rotation samples on a straight spiral
    X = np.column_stack((
        np.cos(seeds)[:, np.newaxis],
        np.sin(seeds)[:, np.newaxis])) * ((1 + seeds + np.pi) / (3 * np.pi) * 3)[:, np.newaxis]

    # Offset each sample along the spiral's normal vector by scaled Gaussian
    # noise
    X += np.column_stack([
        np.cos(seeds_orig),
        np.sin(seeds_orig)]) * (scipy.stats.norm.rvs(size=size) * scipy.stats.norm.pdf(vals))[:, np.newaxis]

    data = torch.from_numpy(X / 2)
    data = data.to(torch.float32)
    return data


# generate pinwheel data
def get_pinwheel(size):
    rng = np.random.RandomState()
    radial_std = 0.3
    tangential_std = 0.1
    num_classes = 5
    num_per_class = size // 5
    rate = 0.25
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = rng.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:, 0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))
    return torch.tensor(2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations)), dtype=torch.float32)


def get_8gauss(size):
    rng = np.random.RandomState()
    scale = 4.
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
               (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                     1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
    centers = [(scale * x, scale * y) for x, y in centers]

    dataset = []
    for i in range(size):
        point = rng.randn(2) * 0.5
        idx = rng.randint(8)
        center = centers[idx]
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)
    dataset = np.array(dataset, dtype="float32")
    dataset /= 1.414
    return torch.tensor(dataset, dtype=torch.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams['image.cmap'] = 'inferno'

    x = get_pinwheel(5000).numpy()
    x1 = x[:, 0]
    x2 = x[:, 1]
    colors = np.arctan2(x2, x1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist2d(x1, x2, range=[[-4, 4], [-4, 4]], bins=100)
    ax.set_aspect('equal', 'box')
    plt.title('Training Samples')
    plt.show()
