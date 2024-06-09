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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import binom
from src.sbc_analysis import get_rank_statistic
from shallow_water_model.simulator import ShallowWaterSimulator as Simulator
from shallow_water_model.prior import DepthProfilePrior as Prior

parser = argparse.ArgumentParser('PCP-Map')
parser.add_argument('--resume', type=str, default="/experiments/sw_100k_64_0.001_3_256_checkpt.pth")
parser.add_argument('--resume50k', type=str, default="/experiments/cond/sw_50k_64_0.001_3_256_checkpt.pth")
parser.add_argument('--resume20k', type=str, default="/experiments/cond/sw_20k_64_0.001_3_256_checkpt.pth")

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


def process_test_data(obs, proj, mean, std, x_dim=100):
    # project observation
    x_star_proj = proj.T @ obs
    # normalize
    x_star_proj_norm = (x_star_proj.T - mean[:, x_dim:]) / std[:, x_dim:]
    return x_star_proj_norm


def generate_theta(generator, x_cond, mean, std, tol, proj_x=None, num_samples=100, printing=True):
    zx = torch.randn(num_samples, 100).to(device)
    if proj_x is not None:
        zx = torch.randn(num_samples, 14).to(device)
    x_cond_tensor = torch.tensor(x_cond, dtype=torch.float32)
    # start sampling timer
    start = time.time()
    x_gen, num_evals = generator.gx(zx, x_cond_tensor.to(device), tol)
    # end timer
    sample_time = time.time() - start
    if printing is True:
        print(f"Sampling Time for theta: {sample_time}")
        print(f"Number of closure calls: {num_evals}")
    if proj_x is not None:
        x_gen = x_gen @ proj_x.T
    theta_gen = x_gen.detach().cpu().numpy()
    # scale back
    theta_gen = (theta_gen * std[:, :100] + mean[:, :100] + 10.0).squeeze()
    return theta_gen


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
    if y_lab is True:
        axis.set_ylabel("Amplitude", rotation=90, fontsize=45)


def plot_prior_predictives(axis, t, x_cond_wonoise, priors, color, y_lab=True, num_samples=50):
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
        axis.plot(x_axs, sim[t, :], c=color, lw=0.3)
    axis.set_xticks([])
    axis.tick_params(axis='y', which='major', labelsize=24)
    if y_lab is True:
        axis.set_ylabel("Amplitude", rotation=90, fontsize=45)
    axis.text(0.1, 0.9, f"t = {t + 1}", transform=axis.transAxes, fontsize=45)


def build_pcpmap(check_point, prior):
    input_x_dim = check_point['args'].input_x_dim
    if bool(check_point['args'].theta_pca) is True:
        input_x_dim = 14
    input_y_dim = check_point['args'].input_y_dim
    feature_dim = check_point['args'].feature_dim
    feature_y_dim = check_point['args'].feature_y_dim
    out_dim = check_point['args'].out_dim
    num_layers_pi = check_point['args'].num_layers_pi
    clip = check_point['args'].clip
    if clip is True:
        reparam = False
    else:
        reparam = True
    picnn = PICNN(input_x_dim, input_y_dim, feature_dim, feature_y_dim, out_dim, num_layers_pi, reparam=reparam)
    pcpmap = PCPMap(prior, picnn)
    pcpmap.load_state_dict(check_point["state_dict_picnn"])
    return pcpmap


def load_data_info(file_path, valid_ratio):
    data = np.load(file_path)['dataset']
    V = np.load(file_path)['Vs']
    trn, _ = train_test_split(data, test_size=valid_ratio, random_state=42)
    mean = np.mean(trn, axis=0, keepdims=True)
    std = np.std(trn, axis=0, keepdims=True)
    train = (trn - mean) / std
    return data, train, V, mean, std


if __name__ == '__main__':

    """Set up PCP-Maps"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    color_list = ['r', 'b', 'salmon']
    time_list = [21, 68, 93]

    # load checkpoints
    checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
    checkpt_50k = torch.load(args.resume50k, map_location=lambda storage, loc: storage)
    checkpt_20k = torch.load(args.resume20k, map_location=lambda storage, loc: storage)
    # build maps
    input_x_dim = checkpt['args'].input_x_dim
    if bool(checkpt['args'].theta_pca) is True:
        input_x_dim = 14
    prior_picnn = distributions.MultivariateNormal(torch.zeros(input_x_dim).to(device), torch.eye(input_x_dim).to(device))
    pcpmap = build_pcpmap(checkpt, prior_picnn)
    pcpmap50k = build_pcpmap(checkpt_50k, prior_picnn)
    pcpmap20k = build_pcpmap(checkpt_20k, prior_picnn)
    pcpmap.to(device)
    pcpmap50k.to(device)
    pcpmap20k.to(device)

    """Grab Training Mean and STD"""
    # TODO change to correct paths
    file_path = '.../PCP-Map/datasets/shallow_water_data3500.npz'
    file_path50k = '.../PCP-Map/datasets/shallow_water_data3500_50k.npz'
    file_path20k = '.../PCP-Map/datasets/shallow_water_data3500_20k.npz'

    dataset, train_data, Vs, train_mean, train_std = load_data_info(file_path, 0.05)
    _, train_data50k, Vs50k, train_mean_50k, train_std_50k = load_data_info(file_path50k, 0.05)
    _, train_data20k, Vs20k, train_mean_20k, train_std_20k = load_data_info(file_path20k, 0.05)
    if bool(checkpt['args'].theta_pca) is True:
        x_full = torch.FloatTensor(train_data[:, :100])
        x_full_50k = torch.FloatTensor(train_data50k[:, :100])
        x_full_20k = torch.FloatTensor(train_data20k[:, :100])
        cov_x = x_full.T @ x_full
        cov_x_50k = x_full_50k.T @ x_full_50k
        cov_x_20k = x_full_20k.T @ x_full_20k
        L, V = torch.linalg.eigh(cov_x)
        L50k, V50k = torch.linalg.eigh(cov_x_50k)
        L20k, V20k = torch.linalg.eigh(cov_x_20k)
        # get the last dx columns in V
        Vx = V[:, -14:].to(device)
        Vx50k = V50k[:, -14:].to(device)
        Vx20k = V20k[:, -14:].to(device)

    """Preparing Plotting Data"""

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
    x_star_processed = process_test_data(x_star_fourier, Vs, train_mean, train_std)
    theta_samples = generate_theta(pcpmap, x_star_processed, train_mean, train_std, checkpt['args'].tol, proj_x=Vx)

    """Ground Truth Plotting"""

    # create plot grid for ground truth values
    fig, axs = plt.subplots(1, 5)
    fig.set_size_inches(40, 7)

    # plot prior samples with ground truth theta
    xx = np.linspace(1, 100, 100)
    axs[0].set_ylim(bottom=4.0, top=18.0)
    axs[0].plot(xx, theta_star.squeeze(0), c='k', linewidth=2)
    for i in range(theta_samples.shape[0]):
        rand_sample = np.random.randint(low=0, high=dataset.shape[0], size=(1,))[0]
        prior_theta_i = dataset[rand_sample, :100] + 10.0
        axs[0].plot(xx, prior_theta_i, c='grey', lw=0.3)
    axs[0].set_xticks([])
    axs[0].tick_params(axis='y', which='major', labelsize=24)
    axs[0].set_ylabel("Depth Profile", rotation=90, fontsize=45)

    # plot 2d ground truth wave image
    img_gt = axs[1].imshow(x_star_nofourier_nonosie, cmap='gray')
    axs[1].axhline(time_list[0], color=color_list[0], linewidth=4)
    axs[1].axhline(time_list[1], color=color_list[1], linewidth=4)
    axs[1].axhline(time_list[2], color=color_list[2], linewidth=4)
    axs[1].set_xticks([])
    axs[1].tick_params(axis='y', which='major', labelsize=24)
    axs[1].margins(0.3)
    axs[1].set_ylabel("Time", rotation=90, fontsize=45)
    axs[1].invert_yaxis()

    # plot prior predictives with ground truth wave
    prior_samples = dataset[:, :100]
    # plot at three times
    plot_prior_predictives(axs[2], time_list[0], x_star_nofourier_nonosie, prior_samples, color=color_list[0])
    plot_prior_predictives(axs[3], time_list[1], x_star_nofourier_nonosie, prior_samples, color=color_list[1], y_lab=False)
    plot_prior_predictives(axs[4], time_list[2], x_star_nofourier_nonosie, prior_samples, color=color_list[2], y_lab=False)

    # save
    fig.tight_layout()
    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_gt_figure.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    """MAP estimation"""

    theta = torch.randn(1, input_x_dim, requires_grad=True).to(device)
    theta_min = theta.clone().detach().requires_grad_(True)
    x_cond_tensor = torch.tensor(x_star_processed, dtype=theta.dtype).to(device)

    def closure():
        loss = -pcpmap.loglik_picnn(theta_min, x_cond_tensor)
        theta_min.grad = torch.autograd.grad(loss, theta_min)[0].detach()
        return loss

    optimizer = torch.optim.LBFGS([theta_min], line_search_fn="strong_wolfe", max_iter=1000000)
    optimizer.step(closure)
    theta_min = (theta_min @ Vx.T).detach().cpu().numpy()
    theta_map = (theta_min * train_std[:, :100] + train_mean[:, :100] + 10.0).squeeze()

    """PCP Posterior Plotting"""

    # create plot grid for ground truth values
    fig, axs = plt.subplots(1, 5)
    fig.set_size_inches(40, 8)

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
    axs[0].set_ylabel("Depth Profile", rotation=90, fontsize=45)

    # plot 2d inferred wave image
    sim_wave = wave_wout_noise(theta_samples[0, :].reshape(1, -1))
    img_sim = axs[1].imshow(sim_wave, cmap='gray')
    axs[1].axhline(time_list[0], color=color_list[0], linewidth=4)
    axs[1].axhline(time_list[1], color=color_list[1], linewidth=4)
    axs[1].axhline(time_list[2], color=color_list[2], linewidth=4)
    axs[1].set_xticks([])
    axs[1].tick_params(axis='y', which='major', labelsize=24)
    axs[1].margins(0.3)
    axs[1].set_ylabel("Time", rotation=90, fontsize=45)
    axs[1].invert_yaxis()

    # plot at three times
    plot_post_predict(axs[2], time_list[0], x_star_nofourier_nonosie, theta_samples, color=color_list[0])
    plot_post_predict(axs[3], time_list[1], x_star_nofourier_nonosie, theta_samples, color=color_list[1], y_lab=False)
    plot_post_predict(axs[4], time_list[2], x_star_nofourier_nonosie, theta_samples, color=color_list[2], y_lab=False)

    # save
    fig.tight_layout()
    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_pcp_figure.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    """Perform SBC Analysis"""

    path_to_test_samps = '.../PCP-Map/datasets/sw_test_data.npz'
    ranks, _ = get_rank_statistic(pcpmap, Vx, train_mean, train_std, checkpt['args'].tol, path_to_test_samps)

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
    fig.tight_layout(pad=3.0)
    spec = fig.add_gridspec(ncols=1, nrows=1)
    ax = fig.add_subplot(spec[0, 0])
    for i in range(ndim):
        hist, *_ = np.histogram(ranks[i], bins=nbins, density=False)
        histcs = hist.cumsum()
        ax.plot(np.linspace(0, nbins, repeats * nbins),
                np.repeat(histcs / histcs.max(), repeats),
                color='r',
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
                    Line2D([0], [0], color='r', lw=1.5, linestyle="-")
                    ]
    ax.legend(custom_lines, ['Uniform CDF', 'PCP-Map'], fontsize=17)

    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_pcp_sbc.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    """Plot PCP from Different Data Size"""

    # sample from posterior
    x_star_processed50k = process_test_data(x_star_fourier, Vs50k, train_mean_50k, train_std_50k)
    theta_samples50k = generate_theta(pcpmap50k, x_star_processed50k, train_mean_50k, train_std_50k,
                                      checkpt_50k['args'].tol, proj_x=Vx50k)
    x_star_processed20k = process_test_data(x_star_fourier, Vs20k, train_mean_20k, train_std_20k)
    theta_samples20k = generate_theta(pcpmap20k, x_star_processed20k, train_mean_20k, train_std_20k,
                                      checkpt_20k['args'].tol, proj_x=Vx20k)

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

    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_pcp_numsims.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, bbox_inches='tight', dpi=300)
    plt.close()
