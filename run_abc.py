from datasets.StochasticLotkaVolterra import StochasticLotkaVolterra
from src.ABC import Rejection_ABC, SMC_ABC
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
# plt.rc('text', usetex=True)
# plt.rc('font', size=12)
import pickle
import scipy.io as sio

# define simulator
LV = StochasticLotkaVolterra()

# define prior bounds
n_dim = 4
logprior_bound_min = -5. * np.ones(n_dim)
logprior_bound_max = 2. * np.ones(n_dim)

# define true observation
xtrue = np.array([0.01, 0.5, 1.0, 0.01])
ytrue = LV.sample_data(xtrue)

# define reference mean and standard deviation
data = sio.loadmat('/ABC/training_data.mat')
ref_mean = np.mean(data['y_train'], axis=0)
ref_std = np.std(data['y_train'], axis=0)

# run rejection ABC
all_x, _, all_eps, all_nsims = SMC_ABC(LV, ytrue, n_dim,
                                       eps_init=10., eps_last=0.5,
                                       n_particles=2000, eps_decay=0.85,
                                       logprior_bound_min=logprior_bound_min,
                                       logprior_bound_max=logprior_bound_max,
                                       ref_mean=ref_mean, ref_std=ref_std)

symbols = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$']
log_limits = [[-5., 2.], [-5., 2.], [-5., 2.], [-5., 2.]]


def plot_matrix(x_samps, limits, xtrue=None, symbols=None):
    dim = x_samps.shape[1]
    plt.figure(figsize=(9, 9))
    for i in range(dim):
        for j in range(i + 1):
            ax = plt.subplot(dim, dim, (i * dim) + j + 1)
            if i == j:
                plt.hist(x_samps[:, i], bins=40, density=True)
                if xtrue is not None:
                    plt.axvline(xtrue[i], color='r', linewidth=3)
                plt.xlim(limits[i])
            else:
                plt.plot(x_samps[:, j], x_samps[:, i], '.k', markersize=.04, alpha=0.1)
                if xtrue is not None:
                    plt.plot(xtrue[j], xtrue[i], '.r', markersize=8, label='Truth')
                # Peform the kernel density estimate
                xlim = limits[j]  # ax.get_xlim()
                ylim = limits[i]  # ax.get_ylim()
                xx, yy = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                kernel = st.gaussian_kde(x_samps[:, [j, i]].T)
                f = np.reshape(kernel(positions), xx.shape)
                ax.contourf(xx, yy, f, cmap='Blues')
                plt.ylim(limits[i])
            plt.xlim(limits[j])
            if symbols is not None:
                if j == 0:
                    plt.ylabel(symbols[i], size=20)
                if i == len(xtrue) - 1:
                    plt.xlabel(symbols[j], size=20)


# plot pair plot for all parameters
for x, eps in zip(all_x, all_eps):
    # plot scatter plots
    plot_matrix(np.log(x), log_limits, xtrue=np.log(xtrue), symbols=symbols)
    plt.gcf().suptitle('$\epsilon = {0:.2}$'.format(eps))
    plt.savefig('StochasticLV_ABCsamples_eps' + "{:2.2f}".format(eps) + '.png')
    # plt.show()
    plt.close()

# save data
data = {"all_x": all_x, "all_eps": all_eps, "all_nsims": all_nsims, "y_true": ytrue}
data_file = 'StochasticLV_ABCsamples.pk'
with open(data_file, "wb") as f:
    pickle.dump(data, f)
