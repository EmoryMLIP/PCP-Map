import os.path
from os import makedirs
from os.path import join
import pickle
import scipy.io
from scipy.stats import binom
import numpy as np
import torch
import torch.nn as nn
import subprocess
import shutil
import time
from importlib import import_module
from shallow_water_model.simulator import ShallowWaterSimulator as Simulator
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

device = "cuda:0"


def _fwd_pass_fourier(profile, seedz):
    _, z = Simulator(outdir=0, fourier=True)(
        profile, seeds_u=[42], seeds_z=[seedz]
    )
    return z


def wave_wout_noise(theta):
    # abs path to solver
    # TODO: change to correct path
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


def plot_post_predict(axis, t, x_cond_wonoise, theta, color, y_lab=True, num_samples=50):
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
        axis.plot(x_axs, sim[t, :], c=color, lw=0.2)
    axis.set_xticks([])
    axis.tick_params(axis='y', which='major', labelsize=24)
    axis.set_xlabel("Position", fontsize=45)
    if y_lab is True:
        axis.set_ylabel("Amplitude", rotation=90, fontsize=45)


def load_data_info(file_path):
    dataset_zip = np.load(file_path)
    data = dataset_zip['dataset']
    V = dataset_zip['V']
    return data, V


# TODO: change to correct path to training data
path_to_data = ".../PCP-Map/datasets/shallow_water_data3500.npz"
def get_rank_statistic(
    generator: nn.Module,
    Vx,
    path_to_samples: str,
    num_samples: int = 1000,
    save: bool = False,
    save_dir: str = None,
):
    sbc = np.load(path_to_samples)
    thos = torch.FloatTensor(sbc["depth_profile"])
    xos = torch.FloatTensor(sbc["z_vals"])[:, :, :, 1:, :]
    Vs = torch.FloatTensor(np.load(path_to_data)['V'])
    thos.to(device)
    xos.to(device)
    Vs.to(device)

    # Calculate ranks
    ndim = thos.shape[-1]
    ranks = [[] for _ in range(ndim)]

    f = torch.distributions.Normal(loc=torch.zeros(1).to(device), scale=10)
    all_samples = []
    for k, (tho, xo) in enumerate(zip(thos.squeeze(), xos.squeeze())):
        xo = (Vs.T @ xo.reshape(-1, 1)).reshape(1, -1)
        samples = generator.sample((num_samples,), x=xo.to(device))
        samples = samples @ Vx.T + 10.0
        all_samples.append(samples.unsqueeze(0).cpu())
        # Calculate rank under Gaussian.
        for i in range(ndim):
            slp = f.log_prob(samples[:, i])
            gtlp = f.log_prob(tho[i]+10.0)
            rr = (slp < gtlp).sum().item()
            ranks[i].append(rr)
    all_samples = np.concatenate(all_samples, 0)
    if save:
        np.savez(join(save_dir, "PCP_SBC.npz"), ranks=ranks, samples=all_samples)
    return np.array(ranks), all_samples


if __name__ == '__main__':

    color_list = ['r', 'b', 'salmon']
    time_list = [21, 68, 93]

    """Load Models"""

    # TODO: change to correct paths
    # load trained posterior
    with open('.../PCP-Map/experiments/npe/sw_NPEpca.p', 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    posterior = content['posterior']
    with open('.../PCP-Map/experiments/npe/sw_NPEpca_50k.p', 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    posterior50k = content['posterior']
    with open('.../PCP-Map/experiments/npe/sw_NPEpca_20k.p', 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    posterior20k = content['posterior']

    """Load Dataset Information"""

    # load ground truth
    ground_truth = scipy.io.loadmat('.../PCP-Map/sw_gt_paper.mat')
    x_star = ground_truth['x_gt']
    theta_star = ground_truth['theta_gt']
    x_star_nofourier_nonosie = wave_wout_noise(theta_star)

    # load data
    # TODO: change to correct paths
    file_path = '.../PCP-Map/datasets/shallow_water_data3500.npz'
    file_path50k = '.../PCP-Map/datasets/shallow_water_data3500_50k.npz'
    file_path20k = '.../PCP-Map/datasets/shallow_water_data3500_20k.npz'
    train_data, Vs = load_data_info(file_path)
    train_data50k, Vs50k = load_data_info(file_path50k)
    train_data20k, Vs20k = load_data_info(file_path20k)
    x_full = train_data[:, :100]
    x_full_50k = train_data50k[:, :100]
    x_full_20k = train_data20k[:, :100]
    cov_x = x_full.T @ x_full
    cov_x_50k = x_full_50k.T @ x_full_50k
    cov_x_20k = x_full_20k.T @ x_full_20k
    L, V = torch.linalg.eigh(torch.tensor(cov_x))
    L50k, V50k = torch.linalg.eigh(torch.tensor(cov_x_50k))
    L20k, V20k = torch.linalg.eigh(torch.tensor(cov_x_20k))
    # get the last dx columns in V
    Vx = V[:, -14:].numpy()
    Vx50k = V50k[:, -14:].numpy()
    Vx20k = V20k[:, -14:].numpy()

    """MAP estimation"""

    # process conditioning variable
    x_star_proj = (Vs.T @ x_star).reshape(1, -1)
    x_cond = torch.tensor(x_star_proj).to(device)
    # start LBFGS for MAP
    theta = torch.randn(1, 14, requires_grad=True).to(device)
    theta_min = theta.clone().detach().requires_grad_(True)

    def closure():
        loss = -posterior.log_prob(theta_min, x_cond, track_gradients=True)
        theta_min.grad = torch.autograd.grad(loss, theta_min)[0].detach()
        return loss

    optimizer = torch.optim.LBFGS([theta_min], line_search_fn="strong_wolfe", max_iter=1000000)
    optimizer.step(closure)
    theta_map = theta_min.detach().cpu().numpy() @ Vx.T + 10.0

    """Posterior Sampling"""

    posterior_samples = posterior.sample((100,), x=x_cond)
    theta_samples = (posterior_samples.detach().cpu().numpy()) @ Vx.T + 10.0

    """NPE Posterior Plotting"""

    # create plot grid for ground truth values
    fig, axs = plt.subplots(1, 5)
    fig.set_size_inches(40, 7)
    xx = np.linspace(1, 100, 100)

    # plot posterior samples with ground truth theta
    axs[0].plot(xx, theta_star.squeeze(0), c='k', linewidth=2)
    axs[0].set_ylim(bottom=4.0, top=18.0)
    # plot map point
    axs[0].scatter(xx, theta_map, c='m', marker='x', s=256)
    for i in range(theta_samples.shape[0]):
        thetai = theta_samples[i, :]
        axs[0].plot(xx, thetai, c='grey', lw=0.2)
    axs[0].set_xticks([])
    axs[0].tick_params(axis='y', which='major', labelsize=24)
    axs[0].set_xlabel("Position", fontsize=45)
    axs[0].set_ylabel("Depth Profile", rotation=90, fontsize=45)

    # plot 2d inferred wave image
    sim_wave = wave_wout_noise(theta_samples[0, :].reshape(1, -1))
    img_sim = axs[1].imshow(sim_wave, cmap='gray')
    axs[1].axhline(time_list[0], color=color_list[0], linewidth=4)
    axs[1].axhline(time_list[1], color=color_list[1], linewidth=4)
    axs[1].axhline(time_list[2], color=color_list[2], linewidth=4)
    axs[1].set_xticks([])
    axs[1].margins(0.3)
    axs[1].set_xlabel("Position", fontsize=45)
    axs[1].set_ylabel("Time", rotation=90, fontsize=45)
    axs[1].tick_params(axis='y', which='major', labelsize=24)
    axs[1].invert_yaxis()

    # plot at three times
    plot_post_predict(axs[2], time_list[0], x_star_nofourier_nonosie, theta_samples, color=color_list[0])
    plot_post_predict(axs[3], time_list[1], x_star_nofourier_nonosie, theta_samples, color=color_list[1], y_lab=False)
    plot_post_predict(axs[4], time_list[2], x_star_nofourier_nonosie, theta_samples, color=color_list[2], y_lab=False)

    # save
    fig.tight_layout()
    sPath = os.path.join('.../PCP-Map/experiments/npe/figs', 'sw_npe_figure.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    """Perform SBC Analysis"""

    # TODO: generate test dataset and change to correct path.
    path_to_test_samps = '.../PCP-Map/datasets/sw_test_data.npz'
    ranks, _ = get_rank_statistic(posterior, torch.FloatTensor(Vx).to(device), path_to_test_samps)
    # plot ranks
    ndim, N = ranks.shape
    nbins = N
    repeats = 1
    hb = binom(N, p=1 / nbins).ppf(0.5) * np.ones(nbins)
    hbb = hb.cumsum() / hb.sum()
    lower = [binom(N, p=p).ppf(0.005) for p in hbb]
    upper = [binom(N, p=p).ppf(0.995) for p in hbb]

    # Plot CDF
    fig = plt.figure(figsize=(8, 5.5))
    spec = fig.add_gridspec(ncols=1, nrows=1)
    ax = fig.add_subplot(spec[0, 0])
    for i in range(ndim):
        hist, *_ = np.histogram(ranks[i], bins=nbins, density=False)
        histcs = hist.cumsum()
        ax.plot(np.linspace(0, nbins, repeats * nbins),
                np.repeat(histcs / histcs.max(), repeats),
                color='m',
                alpha=.1
                )
    ax.plot(np.linspace(0, nbins, repeats * nbins),
            np.repeat(hbb, repeats),
            color="k", lw=2,
            alpha=.8,
            label="uniform CDF")
    ax.fill_between(x=np.linspace(0, nbins, repeats * nbins),
                    y1=np.repeat(lower / np.max(lower), repeats),
                    y2=np.repeat(upper / np.max(lower), repeats),
                    color='k',
                    alpha=.5)
    # Ticks and axes
    ax.set_xticks([0, 500, 1000])
    ax.set_xlim([0, 1000])
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.set_xlabel("Rank", fontsize=20)
    ax.set_yticks([0, .5, 1.])
    ax.set_ylim([0., 1.])
    ax.tick_params(axis='y', which='major', labelsize=16)
    ax.set_ylabel("CDF", fontsize=20)
    # Legend
    custom_lines = [Line2D([0], [0], color="k", lw=1.5, linestyle="-"),
                    Line2D([0], [0], color='m', lw=1.5, linestyle="-")
                    ]
    ax.legend(custom_lines, ['Uniform CDF', 'NPE'], fontsize=17)

    sPath = os.path.join('.../PCP-Map/experiments/npe/figs', 'sw_npe_sbc.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    """Plot NPE from Different Data Size"""

    x_star_proj50k = (Vs50k.T @ x_star).reshape(1, -1)
    x_star_proj20k = (Vs20k.T @ x_star).reshape(1, -1)
    x_cond50k = torch.tensor(x_star_proj50k).to(device)
    x_cond20k = torch.tensor(x_star_proj20k).to(device)
    posterior_samples50k = posterior50k.sample((100,), x=x_cond50k)
    theta_samples50k = (posterior_samples50k.detach().cpu().numpy()) @ Vx50k.T + 10.0
    posterior_samples20k = posterior20k.sample((100,), x=x_cond20k)
    theta_samples20k = (posterior_samples20k.detach().cpu().numpy()) @ Vx20k.T + 10.0

    # grab mean and std
    mean100k = np.mean(theta_samples, axis=0, keepdims=True).squeeze()
    std100k = np.std(theta_samples, axis=0, keepdims=True).squeeze()
    mean50k = np.mean(theta_samples50k, axis=0, keepdims=True).squeeze()
    std50k = np.std(theta_samples50k, axis=0, keepdims=True).squeeze()
    mean20k = np.mean(theta_samples20k, axis=0, keepdims=True).squeeze()
    std20k = np.std(theta_samples20k, axis=0, keepdims=True).squeeze()

    # calculate normed error and plot
    err_100k = np.linalg.norm(theta_star - mean100k) / np.linalg.norm(theta_star)
    err_50k = np.linalg.norm(theta_star - mean50k) / np.linalg.norm(theta_star)
    err_20k = np.linalg.norm(theta_star - mean20k) / np.linalg.norm(theta_star)

    # plot
    font = {'fontname': 'Times'}
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(28, 8)

    xx = np.linspace(1, 100, 100)
    axs[0].plot(xx, theta_star.squeeze(), c='k', label="Ground Truth")
    axs[0].plot(xx, mean20k, c='orange', label="Posterior Mean 20k")
    axs[0].fill_between(xx, (mean20k - std20k), (mean20k + std20k), color='grey', alpha=0.2)
    axs[0].set_ylabel('Depth', fontsize=26)
    axs[0].text(10, 5, f"rel. error = {err_20k:.2f}", fontsize=20, **font)
    axs[0].legend(fontsize="16")

    axs[1].plot(xx, theta_star.squeeze(), c='k', label="Ground Truth")
    axs[1].plot(xx, mean50k, c='b', label="Posterior Mean 50k")
    axs[1].fill_between(xx, (mean50k - std50k), (mean50k + std50k), color='grey', alpha=0.2)
    axs[1].set_xlabel('Position', fontsize=26)
    axs[1].text(10, 5.35, f"rel. error = {err_50k:.2f}", fontsize=20, **font)
    axs[1].legend(fontsize="16")

    axs[2].plot(xx, theta_star.squeeze(), c='k', label="Ground Truth")
    axs[2].plot(xx, mean100k, c='r', label="Posterior Mean 100k")
    axs[2].fill_between(xx, (mean100k - std100k), (mean100k + std100k), color='grey', alpha=0.2)
    axs[2].text(10, 6.45, f"rel. error = {err_100k:.2f}", fontsize=20, **font)
    axs[2].legend(fontsize="16")

    sPath = os.path.join('.../PCP-Map/experiments/npe/figs', 'sw_npe_numsims.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, bbox_inches='tight', dpi=300)
    plt.close()
