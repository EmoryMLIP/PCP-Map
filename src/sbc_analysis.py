from os.path import join
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO change to correct path
path_to_data = ".../PCP-Map/datasets/shallow_water_data3500.npz"


def process_test_data(obs, mean, std, proj_mat, x_dim):
    # project observation
    x_star_proj = proj_mat.T @ obs.cpu().numpy()
    # normalize
    x_star_proj_norm = (x_star_proj.T - mean[:, x_dim:]) / std[:, x_dim:]
    return x_star_proj_norm


def generate_theta(generator, x_cond, mean, std, x_dim, tol, num_samps, proj_x=None):
    zx = torch.randn(num_samps, x_dim).to(device)
    if proj_x is not None:
        zx = torch.randn(num_samps, 14).to(device)
    x_cond_tensor = torch.tensor(x_cond, dtype=torch.float32)
    x_gen, num_evals = generator.gx(zx, x_cond_tensor.to(device), tol)
    if proj_x is not None:
        x_gen = x_gen @ proj_x.T
    theta_gen = x_gen.detach().cpu().numpy()
    # scale back
    theta_gen = (theta_gen * std[:, :x_dim] + mean[:, :x_dim] + 10.0).squeeze()
    return theta_gen


def get_rank_statistic(
    generator: nn.Module,
    Vx,
    trn_mean,
    trn_std,
    tol,
    path_to_samples: str,
    num_samples: int = 1000,
    save: bool = False,
    save_dir: str = None,
):
    """
    Calculate rank statistics.

    generator: trained GATSBI generator network.
    path_to_samples: file from which to load groundtruth samples.
    num_samples: number test samples per conditioning variable.
    save: if True, save ranks as npz file.
    save_dir: location at which to save ranks.
    """
    generator.to(device)
    sbc = np.load(path_to_samples)
    thos = torch.FloatTensor(sbc["depth_profile"])
    xos = torch.FloatTensor(sbc["z_vals"])[:, :, :, 1:, :]
    Vs = np.load(path_to_data)['Vs']

    # Calculate ranks
    ndim = thos.shape[-1]
    ranks = [[] for _ in range(ndim)]

    f = torch.distributions.Normal(loc=torch.zeros(1), scale=10)
    all_samples = []
    for k, (tho, xo) in enumerate(zip(thos.squeeze(), xos.squeeze())):
        xo_processed = process_test_data(xo.reshape(-1, 1), trn_mean, trn_std, Vs, ndim)
        samples = generate_theta(generator, xo_processed, trn_mean, trn_std, ndim, tol, num_samples, proj_x=Vx)
        samples = torch.FloatTensor(samples)
        all_samples.append(samples.unsqueeze(0))
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
