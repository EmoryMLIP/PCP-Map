import numpy as np
import scipy.io
import torch
from tqdm import tqdm
from scipy.stats import multivariate_normal
from torch.distributions import MultivariateNormal
from typing import Optional
from scipy.fft import fft2
from importlib import import_module
import subprocess
import shutil
import time
from os.path import join
from os import makedirs
import emcee

path_to_fcode = '.../PCP-Map/shallow_water_model/shallow_water01_modified.f90'


def gaussian_kernel(size: int, sigma: float, tau: float) -> np.ndarray:
    x = np.arange(size)
    xg, yg = np.meshgrid(x, x)
    sq = (xg - yg) ** 2
    return sigma * np.exp(-sq / (2 * tau))


def spd_gaussian_kernel(gauss_kernel, lamb_min=1e-14):
    return gauss_kernel + 5 * lamb_min * np.eye(100)


# define posterior class
class Posterior:
    def __init__(self, prior, likelihood):
        self.prior = prior
        self.likelihood = likelihood
        self.dim = self.prior.dim

    def logpdf(self, x):
        return self.prior.logpdf(x) + self.likelihood.logpdf(x)


# define Gaussian rv proposal
class GaussianProposal:
    def __init__(self, cov):
        self.cov = cov
        self.cov_spd = spd_gaussian_kernel(cov)

    def sample(self, x):
        return multivariate_normal.rvs(mean=x, cov=self.cov)

    def logpdf(self, xp, x):
        logprob = MultivariateNormal(torch.tensor(x), torch.tensor(self.cov_spd)).log_prob(torch.tensor(xp))
        return logprob.item()


# define samplers
class MetropolisHastingsSampler:

    def __init__(self, pi, prop):
        # check definitions of pi and prop
        assert hasattr(pi, 'logpdf') and hasattr(prop, 'logpdf'), 'define classes with logpdf function'
        # assign inputs
        self.pi = pi
        self.prop = prop

    def sample(self, x0, n_steps: int, bounds=None):
        """ Sample from target density pi using a MCMC chain of
            length n_samps starting from x0 """

        # define array to store samples
        dim = x0.size
        samps = np.zeros((n_steps + 1, dim))
        logpdfs = np.zeros((n_steps + 1,))

        # define bounds and check dimensions
        if np.any(bounds is None):
            bounds = np.array([[-np.inf] * dim, [np.inf] * dim])
        assert (bounds.shape == (2, dim))

        # define counter of accepted samples
        n_accept = 0

        # define xold at x0
        xold = x0
        samps[0, :] = xold

        # evaluate target at xold
        logpdf_old = self.pi.logpdf(xold)
        logpdfs[0] = logpdf_old

        for i in tqdm(range(1, n_steps + 1), ascii=True, ncols=100):

            # sample from proposal
            xnew = self.propose_sample(xold, samps, i)

            # check if sample is inside bounds
            if any(xnew < bounds[0, :]) or any(xnew > bounds[1, :]):
                logpdf_new = -np.inf
            else:
                logpdf_new = self.pi.logpdf(xnew)

            # evaluate density under proposal
            AcceptProb = self.acceptance_probability(xold, xnew, logpdf_old, logpdf_new)

            # accept or reject samle
            if np.random.random() < AcceptProb:
                xold = xnew
                logpdf_old = logpdf_new
                n_accept += 1

            # save sample
            samps[i, :] = xold
            logpdfs[i] = logpdf_old

        # print acceptance probability
        accept_rate = float(n_accept) / float(n_steps)
        print("Overall acceptance rate: %3.1f\n" % (accept_rate * 100))

        # return samples
        return samps, logpdfs, accept_rate

    def propose_sample(self, xold, samps, iteration):
        return self.prop.sample(xold)

    def acceptance_probability(self, xold, xnew, logpdf_old, logpdf_new):
        # evaluate proposal density under proposal
        logprop_old = self.prop.logpdf(xold, xnew)
        logprop_new = self.prop.logpdf(xnew, xold)
        logdetBalance = logpdf_new - logpdf_old + logprop_old - logprop_new
        return np.min((1., np.exp(logdetBalance)))


class AdaptiveMetropolisSampler(MetropolisHastingsSampler):

    def __init__(self, pi, prop, min_iter_adapt=100, iter_step_adapt=5):
        # call parent init
        super(AdaptiveMetropolisSampler, self).__init__(pi, prop)
        # assign adaptation parameters
        self.min_iter_adapt = min_iter_adapt
        self.iter_step_adapt = iter_step_adapt
        assert isinstance(prop, GaussianProposal)

    def propose_sample(self, xold, samps, iteration):
        # adapt proposal
        if (iteration > self.min_iter_adapt) and np.mod(iteration, self.iter_step_adapt):
            emp_pert = samps[:iteration, :] - np.mean(samps[:iteration, :], axis=0)
            emp_cov = np.dot(emp_pert.T, emp_pert) / (iteration - 1)
            self.prop.cov = emp_cov + 1e-6 * np.eye(emp_cov.shape[0])
        # sample proposal
        return self.prop.sample(xold)


class PreConditionedCrankNicolsonSampler(MetropolisHastingsSampler):

    def __init__(self, pi, prop, beta=0.8):
        # call parent init
        super(PreConditionedCrankNicolsonSampler, self).__init__(pi, prop)
        # set beta and proposal covariance
        self.beta = beta
        self.prop.cov *= self.beta ** 2
        self.prop.cov_spd = spd_gaussian_kernel(self.prop.cov)
        # confirm that pi is a posterior with prior + likelihood
        assert isinstance(prop, GaussianProposal)
        assert isinstance(pi, Posterior)

    def propose_sample(self, x_old, samps, iteration):
        # set proposal covariance
        return self.prop.sample(np.sqrt(1.0 - self.beta ** 2) * x_old)

    def acceptance_probability(self, xold, xnew, *_):
        logdetBalance = self.pi.likelihood.logpdf(xnew) - self.pi.likelihood.logpdf(xold)
        return np.min((1., np.exp(logdetBalance)))


if __name__ == '__main__':

    """Define Posterior Density"""

    class prior:
        def __init__(
            self,
            size: Optional[int] = 100,
            sigma: Optional[float] = 15.0,
            tau: Optional[float] = 100.0,
            mean: Optional[float] = 10.0
        ):
            self.size = size
            self.mean = mean
            self.cov = gaussian_kernel(size, sigma, tau)
            self.cov_spd = spd_gaussian_kernel(self.cov)
            self.dim = self.cov.shape[0]

        def logpdf(self, theta):
            loc = np.ones(self.size) * self.mean
            logprob = MultivariateNormal(torch.tensor(loc), torch.tensor(self.cov_spd)).log_prob(torch.tensor(theta))
            return logprob.item()


    class sw_likelihood:
        def __init__(self, x_cond, proj):
            self.dim = proj.shape[1]
            self.x = x_cond
            self.proj = proj
            self.proj_mat = proj.T @ proj
            try:
                self.sw = import_module("shallow_water")
            except ModuleNotFoundError:
                bashcommand = "python -m numpy.f2py -c %s -m shallow_water" % path_to_fcode
                subprocess.call(bashcommand.split(" "))
                self.sw = import_module("shallow_water")

        def logpdf(self, theta):
            # set up temporary save dir
            outdir = int((time.time() % 1) * 1e7)
            makedirs("%07d" % outdir, exist_ok=True)
            file_z = join("%07d" % outdir, "z%s.dat")

            # obtain f(theta)
            self.sw.shallow_water(theta, int(outdir))
            z = np.zeros((101, 100))
            for i in range(0, 101):
                str_i = ("{0:03d}").format(i)
                with open(file_z % (str_i), "r") as f:
                    z[i] = np.loadtxt(f)

            # perform FFT
            fft_z_real = np.expand_dims(fft2(z).real, 0)
            fft_z_imag = np.expand_dims(fft2(z).imag, 0)
            z = np.concatenate([fft_z_real, fft_z_imag], 0)

            # project data
            z_vals = z[:, 1:, :]
            z_vals = z_vals.reshape(-1, 1)
            z_vals_proj = self.proj.T @ z_vals

            shutil.rmtree("%07d" % outdir)

            # compute new covariance
            covar = 0.25 ** 2 * self.proj_mat
            # calculate log probability
            logprob = MultivariateNormal(torch.tensor(z_vals_proj.reshape(-1)), torch.tensor(covar)).log_prob(torch.tensor(self.x))
            return logprob.item()

    """Algorithm Starts"""

    # define posterior
    # TODO change to ground truth path
    dataset = scipy.io.loadmat('.../PCP-Map/sw_gt_paper.mat')
    x_star = dataset['x_gt']
    Vs = np.load(".../PCP-Map/datasets/shallow_water_data3500.npz")['Vs']
    x_star_proj = (Vs.T @ x_star).reshape(-1)
    pi = Posterior(prior(), sw_likelihood(x_cond=x_star_proj, proj=Vs))

    # set initial condition
    theta0 = np.random.randn(100)

    # define proposal for adaptive metropolis and run sampler
    import copy
    cov = gaussian_kernel(100, 15.0, 100.0)
    prop = GaussianProposal(copy.copy(cov))

    # define tuning parameters
    alpha_min = 0.3
    alpha_max = 0.5
    beta_factor = 1.05

    # run to remove burn-in
    beta = 0.4
    pCN = PreConditionedCrankNicolsonSampler(pi, prop, beta=beta)
    samps, _, accept_rate = pCN.sample(theta0, 5000)
    theta0 = samps[-1, :]

    # tune the beta parameter to get an acceptance rate between 0.3 and 0.4
    accept_rate = np.inf
    while (1):
        # run pCN and compute acceptance rate
        print('Beta = ' + str(beta))
        prop = GaussianProposal(copy.copy(cov))
        pCN = PreConditionedCrankNicolsonSampler(pi, prop, beta=beta)
        samps, _, accept_rate = pCN.sample(theta0, 5000)
        # re-initialize based on last chain
        theta0 = samps[-1, :]
        # update beta based on acceptance rate
        if accept_rate < alpha_min:
            beta = beta / beta_factor
        elif accept_rate > alpha_max:
            beta = beta * beta_factor
        else:
            break

    # run final run with optimal acceptance
    pCN = PreConditionedCrankNicolsonSampler(pi, prop, beta=beta)
    pCN_samps, pCN_logpdfs, accept_rate = pCN.sample(theta0, 100000)
    # calculated marginal-averaged ESS
    rho_avg = np.mean([emcee.autocorr.integrated_time(pCN_samps[1:, k]) for k in np.arange(100)])
    ess = 100000 / (1 + 2*rho_avg)

    # save results
    makedirs('mcmc_sampled_data', exist_ok=True)
    outfile = join('.../PCP-Map/mcmc_sampled_data', "pCNmcmc_data.npz")
    np.savez_compressed(
        outfile,
        theta=pCN_samps,
        logpdf=pCN_logpdfs,
        accept_rate=accept_rate,
        ESS=ess
    )
    print(f"The ESS is {ess}")
