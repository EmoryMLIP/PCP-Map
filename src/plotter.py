import os
import datasets
from datasets import tabular_data
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno'


def plot4_tabular(dataset, z_gen, sample_gen, sPath, sTitle="", hidevals=True):
    """
    :param dataset: data set name
    :param z_gen: pulled back sample
    :param sample_gen: generated sample
    :param sPath: save path for figures
    :param sTitle: save title for figures
    :param hidevals: hide axes values
    """
    # convert
    z = z_gen.detach().cpu().numpy()
    sample = sample_gen.detach().cpu().numpy()

    if dataset == 'rd_wine' or dataset == 'wt_wine':    # plot dim 5 and 7
        d1 = 4
        d2 = 6
    elif dataset == 'parkinson':    # plot dim 6 and 14
        d1 = 5
        d2 = 13
    else:
        raise Exception("Dataset is Incorrect")

    nBins = 100
    LOWX = -4
    HIGHX = 4
    LOWY = -4
    HIGHY = 4

    # get training sample for visualization
    if dataset == 'rd_wine':
        sample_plot = tabular_data.get_rd_wine()
    elif dataset == 'wt_wine':
        sample_plot = tabular_data.get_wt_wine()
    elif dataset == 'parkinson':
        sample_plot = tabular_data.get_parkinson()

    # remove correlated columns
    if dataset in ['rd_wine', 'wt_wine', 'parkinson']:
        sample_plot = tabular_data.process_data(sample_plot)
        sample_plot = tabular_data.normalize_data(sample_plot)

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(21, 7)
    fig.suptitle(sTitle)

    im1, _, _, map1 = axs[0].hist2d(z[:, d1], z[:, d2], range=[[-6, 6], [-6, 6]], bins=nBins)
    axs[0].set_title('f(x)')
    im2, _, _, map2 = axs[1].hist2d(sample_plot[:, d1], sample_plot[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins)
    axs[1].set_title('x from pi')
    im3, _, _, map3 = axs[2].hist2d(sample[:, d1], sample[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins)
    axs[2].set_title('finv(z)')

    if hidevals is True:
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[2].set_xticks([])
        axs[2].set_yticks([])

    fig.colorbar(map1, cax=fig.add_axes([0.353, 0.11, 0.01, 0.77]))
    fig.colorbar(map2, cax=fig.add_axes([0.624, 0.11, 0.01, 0.77]))
    fig.colorbar(map3, cax=fig.add_axes([0.90,  0.11, 0.01, 0.77]))

    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()


def plot_matrix(x_samps, limits, xtrue=None, xmap=None, symbols=None):
    dim = x_samps.shape[1]
    plt.figure(figsize=(9, 9))
    for i in range(dim):
        for j in range(i+1):
            ax = plt.subplot(dim, dim, (i*dim)+j+1)
            if i == j:
                plt.hist(x_samps[:, i], bins=40, density=True)
                if xtrue is not None:
                    plt.axvline(xtrue[i], color='r', linewidth=2)
                if xmap is not None:
                    plt.axvline(xmap[i], color='k', linewidth=2)
                plt.xlim(limits[i])
                if i != 0:
                    ax.set_yticklabels([])
                if i != 3:
                    ax.set_xticklabels([])
            else:
                plt.plot(x_samps[:, j], x_samps[:, i], '.k', markersize=.04, alpha=0.1)
                if xtrue is not None:
                    plt.plot(xtrue[j], xtrue[i], '.r', markersize=7, label='Truth')
                if xmap is not None:
                    plt.plot(xmap[j], xmap[i], 'xk', markersize=7, label='Truth')
                if i < 3:
                    ax.set_xticklabels([])
                if j == 1 or j == 2:
                    ax.set_yticklabels([])
                # Peform the kernel density estimate
                xlim = limits[j]   #ax.get_xlim()
                ylim = limits[i]   #ax.get_ylim()
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
                if i == len(xtrue)-1:
                    plt.xlabel(symbols[j], size=20)
