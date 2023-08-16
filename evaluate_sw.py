import argparse
import scipy.io
import os
import torch
import subprocess
import shutil
import time
from importlib import import_module
from os import makedirs
from os.path import join
import numpy as np
from sklearn.model_selection import train_test_split
from torch import distributions
from src.icnn import PICNN
from src.pcpmap import PCPMap
from datasets import shallow_water
import matplotlib.pyplot as plt
from shallow_water_model.simulator import ShallowWaterSimulator as Simulator
from shallow_water_model.prior import DepthProfilePrior as Prior

parser = argparse.ArgumentParser('PCP-Map')
parser.add_argument('--resume', type=str, default="/experiments/tabcond/sw/...")

args = parser.parse_args()


def _fwd_pass_fourier(profile, seedz):
    _, z = Simulator(outdir=0, fourier=True)(
        profile, seeds_u=[42], seeds_z=[seedz]
    )
    return z


def wave_wout_noise(theta):
    # abs path to solver
    path_to_fcode = '.../PCP-Map/shallow_water_model/shallow_water01_modified.f90'
    # load solver
    try:
        sw = import_module("shallow_water")
    except ModuleNotFoundError:
        bashcommand = "python -m numpy.f2py -c %s -m shallow_water" % path_to_fcode
        subprocess.call(bashcommand.split(" "))
        sw = import_module("shallow_water")
    # set up temporary dir and file
    outdir = int((time.time() % 1) * 1e7)
    makedirs("%07d" % outdir, exist_ok=True)
    file_z = join("%07d" % outdir, "z%s.dat")
    # simulate wave
    sw.shallow_water(theta, int(outdir))
    # read z output into single array
    z = np.zeros((101, 100))
    for i in range(0, 101):
        str_i = "{0:03d}".format(i)
        with open(file_z % str_i, "r") as f:
            z[i] = np.loadtxt(f)

    # Remove save directory to free memory
    shutil.rmtree("%07d" % outdir)
    return z[1:, :]


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
    theta_gen = (theta_gen * std[:, :x_dim] + mean[:, :x_dim] + 10).squeeze()
    return theta_gen


def plot_post_predict(axis, t, x_cond_wonoise, theta, y_lab=True, num_samples=50):
    x_axs = np.linspace(1, 100, 100)
    # plot ground truth at time t
    axis.plot(x_axs, x_cond_wonoise[t, :], c='k')
    # plot posterior predictives using num_samples random samples
    for _ in range(num_samples):
        rand_sample = np.random.randint(low=0, high=theta.shape[0], size=(1,))[0]
        theta_i = theta[rand_sample, :]
        theta_i = np.expand_dims(theta_i, 0)
        # run forward model
        sim = wave_wout_noise(theta_i)
        # plot simulated wave at time t
        axis.plot(x_axs, sim[t, :], c='r', lw=0.2)
    axis.set_xticks([])
    if y_lab is True:
        axis.set_ylabel("Amplitude", rotation=90, fontsize=24)


def plot_prior_predictives(axis, t, x_cond_wonoise, priors, y_lab=True, num_samples=50):
    x_axs = np.linspace(1, 100, 100)
    # plot ground truth at time t
    axis.plot(x_axs, x_cond_wonoise[t, :], c='k')
    # plot prior predictives using num_samples random samples
    for _ in range(num_samples):
        rand_sample = np.random.randint(low=0, high=priors.shape[0], size=(1,))[0]
        priors_i = priors[rand_sample, :]
        priors_i = np.expand_dims(priors_i, 0)
        # run forward model
        sim = wave_wout_noise(priors_i)
        # plot simulated wave at time t
        axis.plot(x_axs, sim[t, :], c='grey', lw=0.3)
    axis.set_xticks([])
    if y_lab is True:
        axis.set_ylabel("Amplitude", rotation=90, fontsize=24)
    axis.text(0.1, 0.9, f"t = {t+1}", transform=axis.transAxes, fontsize=20)


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

    """
    Preparing Plotting Data
    """

    # sample for ground truth prior
    seed_depth = 77777
    theta_star = Prior(return_seed=False)(seed=seed_depth)

    # obtain x_star=f(theta_star)
    x_fourier = _fwd_pass_fourier(theta_star, seedz=seed_depth)
    x_vals_fourier = x_fourier.squeeze()
    x_vals_fourier = x_vals_fourier[:, 1:, :]
    x_star_fourier = x_vals_fourier.reshape(-1, 1)

    # obtain noiseless wave from theta_star
    x_star_nofourier_nonosie = wave_wout_noise(theta_star)

    # save ground truth values
    file_name = f"sw_gt{seed_depth}.mat"
    scipy.io.savemat(file_name, {'theta_gt': theta_star, 'x_gt': x_star_fourier, 'wave_gt': x_star_nofourier_nonosie})

    # generate theta from PCP-Map
    x_star_processed = process_test_data(x_star_fourier, train_mean, train_std, input_x_dim)
    theta_samples = generate_theta(pcpmap, x_star_processed, train_mean, train_std, input_x_dim, checkpt['args'].tol)

    """
    Ground Truth Plotting
    """

    # create plot grid for ground truth values
    fig, axs = plt.subplots(1, 5)
    fig.set_size_inches(40, 8)

    # plot prior samples with ground truth theta
    xx = np.linspace(1, 100, 100)
    axs[0].plot(xx, theta_star.squeeze(0), c='k')
    for i in range(theta_samples.shape[0]):
        rand_sample = np.random.randint(low=0, high=dataset.shape[0], size=(1,))[0]
        prior_theta_i = dataset[rand_sample, :input_x_dim] + 10.0
        axs[0].plot(xx, prior_theta_i, c='grey', lw=0.3)
    axs[0].set_xticks([])
    axs[0].set_ylabel("Depth Profile", rotation=90, fontsize=24)

    # plot 2d ground truth wave image
    img_gt = axs[1].imshow(x_star_nofourier_nonosie, cmap='gray')
    axs[1].set_xticks([])
    axs[1].margins(0.3)
    axs[1].set_ylabel("Time", rotation=90, fontsize=24)
    axs[1].invert_yaxis()

    # plot prior predictives with ground truth wave
    time_list = [21, 68, 93]
    prior_samples = dataset[:, :input_x_dim]
    # plot at three times
    plot_prior_predictives(axs[2], time_list[0], x_star_nofourier_nonosie, prior_samples)
    plot_prior_predictives(axs[3], time_list[1], x_star_nofourier_nonosie, prior_samples, y_lab=False)
    plot_prior_predictives(axs[4], time_list[2], x_star_nofourier_nonosie, prior_samples, y_lab=False)

    # save
    fig.tight_layout()
    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_gt_figure.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    """
    PCP Posterior Plotting
    """

    # create plot grid for ground truth values
    fig, axs = plt.subplots(1, 5)
    fig.set_size_inches(40, 8)

    # plot posterior samples with ground truth theta
    axs[0].plot(xx, theta_star.squeeze(0), c='k')
    for i in range(theta_samples.shape[0]):
        thetai = theta_samples[i, :]
        axs[0].plot(xx, thetai, c='r', lw=0.2)
    axs[0].set_xticks([])
    axs[0].set_ylabel("Depth Profile", rotation=90, fontsize=24)

    # plot 2d inferred wave image
    sim_wave = wave_wout_noise(theta_samples[0, :].reshape(1, -1))
    img_sim = axs[1].imshow(sim_wave, cmap='Reds')
    axs[1].set_xticks([])
    axs[1].margins(0.3)
    axs[1].set_ylabel("Time", rotation=90, fontsize=24)
    axs[1].invert_yaxis()

    # plot posterior predictives with ground truth wave
    time_list = [21, 68, 93]
    # plot at three times
    plot_post_predict(axs[2], time_list[0], x_star_nofourier_nonosie, theta_samples)
    plot_post_predict(axs[3], time_list[1], x_star_nofourier_nonosie, theta_samples, y_lab=False)
    plot_post_predict(axs[4], time_list[2], x_star_nofourier_nonosie, theta_samples, y_lab=False)

    # save
    fig.tight_layout()
    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_pcp_figure.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    """Evaluate Ground Truth NLL"""

    x_test = torch.tensor((theta_star.reshape(1, -1) - train_mean[:, :100] - 10) / train_std[:, :100],
                          dtype=torch.float32).to(device)
    y_test = torch.tensor(x_star_processed, dtype=torch.float32).reshape(1, -1).to(device)
    x_test.requires_grad_(True)
    y_test.requires_grad_(True)
    print("Ground Truth NLL: " + str(-pcpmap.loglik_picnn(x_test, y_test).mean().item()))
