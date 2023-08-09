import argparse
import os
import time
import torch
from typing import Optional
import numpy as np
from sklearn.model_selection import train_test_split
from torch import distributions
from src.icnn import PICNN
from src.pcpmap import PCPMap
from datasets import shallow_water
import matplotlib.pyplot as plt
from datasets.shallow_water_model.simulator import ShallowWaterSimulator as Simulator
from datasets.shallow_water_model.prior import DepthProfilePrior as Prior


parser = argparse.ArgumentParser('PCP-Map')
parser.add_argument('--resume',    type=str, default="/experiments/tabcond/sw_fourier/swPCA/...")

args = parser.parse_args()


def observation_noise(
    observation: np.ndarray,
    seed: Optional[int] = 42,
    gain: Optional[float] = 1,
    scale: Optional[float] = 0.25,
) -> np.ndarray:
    """
    Add white noise to observations.

    Args:
        observation: simulation to which to add noise.
        seed: random-sampling seed.
        gain: gain value to scale up observation values.
        scale: std of white noise.
    """
    np.random.seed(seed)
    return gain * (observation) + (scale * np.random.randn(*observation.shape))


def _seed_by_time_stamp(num_seeds):
    seeds = []
    for i in range(num_seeds):
        tic = time.time()
        seeds.append(int((tic % 1) * 1e7))
    return seeds


def _fwd_pass_fourier(profile):
    seed_depth, seed_u, seed_z = _seed_by_time_stamp(3)
    tic = time.time()
    _, z = Simulator(outdir=seed_depth, fourier=True)(
        profile, seeds_u=[seed_u], seeds_z=[seed_z]
    )
    return z


def _fwd_pass_nofourier(profile):
    seed_depth, seed_u, seed_z = _seed_by_time_stamp(3)
    _, z = Simulator(outdir=seed_depth, fourier=False)(
        profile, seeds_u=[seed_u], seeds_z=[seed_z]
    )
    return z


def process_test_data(obs, mean, std, x_dim):
    # project observation
    Vs = shallow_water.create_data_swe(num_eigs=3500, save=False).cpu().numpy()
    x_star_proj = Vs.T @ obs
    # normalize
    x_star_proj_norm = (x_star_proj.T - mean[:, x_dim:]) / std[:, x_dim:]
    return x_star_proj_norm


def generate_theta(generator, x_cond, mean, std, x_dim, tol, num_samples=100):

    zx = torch.randn(num_samples, 100).to(device)
    x_cond_tensor = torch.tensor(x_cond, dtype=torch.float32)
    # start sampling timer
    start = time.time()
    x_gen, num_evals = generator.gx(zx, x_cond_tensor.to(device), tol)
    # end timer
    sample_time = time.time() - start
    print(f"Sampling Time for theta: {sample_time}")
    print(f"Number of closure calls: {num_evals}")
    theta_gen = x_gen.detach().cpu().numpy()
    # scale back
    theta_gen = (theta_gen * std[:, :x_dim] + mean[:, :x_dim]+10).squeeze()
    return theta_gen


def plot_post_predict(axis, t, x_cond, theta, num_samples=10):
    x_axs = np.linspace(0, 100, 100)
    # plot ground truth at time t
    axis.plot(x_axs, x_cond[t, :], c='k')
    # plot posterior predictives using num_samples random samples
    for _ in range(num_samples):
        rand_sample = np.random.randint(low=0, high=theta.shape[0], size=(1,))[0]
        theta_i = theta[rand_sample, :]
        theta_i = np.expand_dims(theta_i, 0)
        # run forward model
        sim = _fwd_pass_nofourier(theta_i)
        sim_vals = sim.squeeze()
        sim_vals = sim_vals[1:, :]
        # plot simulated wave at time t
        axis.plot(x_axs, sim_vals[t, :], c='r', lw=0.2)


def plot_prior_predictives(axis, t, x_cond, priors, num_samples=10):
    x_axs = np.linspace(0, 100, 100)
    # plot ground truth at time t
    axis.plot(x_axs, x_cond[t, :], c='k')
    # plot prior predictives using num_samples random samples
    for _ in range(num_samples):
        rand_sample = np.random.randint(low=0, high=priors.shape[0], size=(1,))[0]
        priors_i = priors[rand_sample, :]
        priors_i = np.expand_dims(priors_i, 0)
        # run forward model
        sim = _fwd_pass_nofourier(priors_i)
        sim_vals = sim.squeeze()
        sim_vals = sim_vals[1:, :]
        # plot simulated wave at time t
        axis.plot(x_axs, sim_vals[t, :], c='b', lw=0.2)


if __name__ == '__main__':

    """Set up PCP-Map"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)

    input_x_dim = checkpt['args'].input_x_dim
    input_y_dim = checkpt['args'].input_y_dim
    feature_dim = checkpt['args'].feature_dim
    feature_y_dim = checkpt['args'].feature_y_dim
    out_dim = checkpt['args'].out_dim
    num_layers_pi = checkpt['args'].num_layers_pi
    clip = checkpt['args'].clip
    if clip is True:
        reparam = False
    else:
        reparam = True

    prior_picnn = distributions.MultivariateNormal(torch.zeros(input_x_dim).to(device), torch.eye(input_x_dim).to(device))
    picnn = PICNN(input_x_dim, input_y_dim, feature_dim, feature_y_dim, out_dim, num_layers_pi, reparam=reparam).to(device)
    pcpmap = PCPMap(prior_picnn, picnn)
    pcpmap.load_state_dict(checkpt["state_dict_picnn"])
    pcpmap = pcpmap.to(device)

    """Grab Training Mean and STD"""

    dataset, _, _, _ = shallow_water.load_swdata(checkpt['args'].batch_size)
    train, valid = train_test_split(
        dataset,
        test_size=0.001,
        random_state=42
    )
    train_mean = np.mean(train, axis=0, keepdims=True)
    train_std = np.std(train, axis=0, keepdims=True)

    """Sample and Plotting"""

    seed_depth, _, _ = _seed_by_time_stamp(3)
    theta_star = Prior(return_seed=False)(seed=seed_depth)

    # obtain x_star=f(theta_star)
    x_fourier = _fwd_pass_fourier(theta_star)
    x_vals_fourier = x_fourier.squeeze()
    x_vals_fourier = x_vals_fourier[:, 1:, :]
    x_star_fourier = x_vals_fourier.reshape(-1, 1)

    # generate samples from PCP-Map
    x_star_processed = process_test_data(x_star_fourier, train_mean, train_std, input_x_dim)
    theta_samples = generate_theta(pcpmap, x_star_processed, train_mean, train_std, input_x_dim, checkpt['args'].tol)

    # Plot compare posterior samples
    xx = np.linspace(0, 100, 100)
    plt.plot(xx, theta_star.squeeze(0), c='k')
    # plt.plot(xx, theta_star, c='k')
    for i in range(theta_samples.shape[0]):
        thetai = theta_samples[i, :]
        plt.plot(xx, thetai, c='r', lw=0.2)
    # save
    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_posterior_samples.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    # Plot compare posterior predictives
    time_list = [21, 68, 93]
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(21, 7)

    x_nofourier = _fwd_pass_nofourier(theta_star)
    x_vals_nofourier = x_nofourier.squeeze()
    x_star_nofourier = x_vals_nofourier[1:, :]
    # x_star_nofourier = x_star

    # plot at three times
    plot_post_predict(axs[0], time_list[0], x_star_nofourier, theta_samples)
    plot_post_predict(axs[1], time_list[1], x_star_nofourier, theta_samples)
    plot_post_predict(axs[2], time_list[2], x_star_nofourier, theta_samples)
    # save
    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_posterior_predicts.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    # Plot compare prior predictives
    time_list = [21, 68, 93]
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(21, 7)
    # plot at three times
    plot_prior_predictives(axs[0], time_list[0], x_star_nofourier, dataset[:, :input_x_dim])
    plot_prior_predictives(axs[1], time_list[1], x_star_nofourier, dataset[:, :input_x_dim])
    plot_prior_predictives(axs[2], time_list[2], x_star_nofourier, dataset[:, :input_x_dim])
    # save
    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_prior_predicts.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()
