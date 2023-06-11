import argparse
import scipy.io
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import distributions
from src.icnn import PICNN
from src.plotter import plot_matrix
from src.triflow_picnn import TriFlowPICNN
from datasets.StochasticLotkaVolterra import StochasticLotkaVolterra
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('PCPM')
parser.add_argument('--resume',    type=str, default="/experiments/tabcond/lv/lv_2023_04_12_14_52_05_64_0.005_5_512_checkpt.pth")

args = parser.parse_args()


def experiment(LV, abc_dat_path, theta_star, flow):

    # grab y star from ABC
    abc_sample = pd.read_pickle(abc_dat_path)
    y_theta_star = abc_sample["y_true"]
    y_theta_star_norm = (y_theta_star - train_mean[:, input_x_dim:]) / train_std[:, input_x_dim:]
    y_theta_star_norm_tensor = torch.tensor(y_theta_star_norm, dtype=torch.float32)

    # generate samples
    zx = torch.randn(2000, 4).to(device)
    x_gen, num_evals = flow.g2(zx, y_theta_star_norm_tensor.to(device), checkpt['args'].tol)
    x_gen = x_gen.detach().to(device)
    print("Number of closure calls: " + str(num_evals))
    theta_gen = x_gen.detach().cpu().numpy()
    theta_gen = (theta_gen * train_std[:, :input_x_dim] + train_mean[:, :input_x_dim]).squeeze()

    # plot
    theta_star_log = np.log(theta_star)
    symbols = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$']
    log_limits = [[-5., 2.], [-5., 2.], [-5., 2.], [-5., 2.]]
    plot_matrix(theta_gen, log_limits, xtrue=theta_star_log, symbols=symbols)
    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_' + str(theta_star[0].item()) + '.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    # plot posterior predictive
    plt.figure()
    ytrue = LV.simulate(theta_star)
    c1 = plt.plot(LV.tt, ytrue[:, 0], '-', label='Predators')
    c2 = plt.plot(LV.tt, ytrue[:, 1], '-', label='Prey')
    for i in range(10):
        rand_sample = np.random.randint(low=0, high=2000, size=(1,))[0]
        xi = np.exp(theta_gen[rand_sample, :])
        yt = LV.simulate(xi)
        plt.plot(LV.tt, yt[:, 0], '--', color=c1[0].get_color(), alpha=0.3)
        plt.plot(LV.tt, yt[:, 1], '--', color=c2[0].get_color(), alpha=0.3)
    plt.xlabel('$t$', size=20)
    plt.ylabel('$Z(t)$', size=20)
    plt.legend(loc='upper right')
    plt.xlim(0, 20)
    sPath = os.path.join(checkpt['args'].save, 'figs',
                         checkpt['args'].data + '_' + str(theta_star[0].item()) + '_post.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()


if __name__ == '__main__':

    """
    Load Best Model
    """

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
    flow_picnn = TriFlowPICNN(prior_picnn, picnn)
    flow_picnn.load_state_dict(checkpt["state_dict_picnn"])
    flow_picnn = flow_picnn.to(device)

    """
    Test Generated Sample
    """

    # dataset_load = scipy.io.loadmat('.../PCPM/datasets/training_data500k.mat')    # load 500k training samples
    # dat = dataset_load["train_data"]
    dataset_load = scipy.io.loadmat('.../PCPM/datasets/training_data.mat')
    x_train = dataset_load['x_train']
    y_train = dataset_load['y_train']
    dat = np.concatenate((x_train, y_train), axis=1)
    # log transformation over theta
    dat[:, :4] = np.log(dat[:, :4])

    train, valid = train_test_split(
            dat, test_size=0.10,
            random_state=42
        )
    train_mean = np.mean(train, axis=0, keepdims=True)
    train_std = np.std(train, axis=0, keepdims=True)

    StochLV = StochasticLotkaVolterra()
    path_theta1 = '.../PCPM/experiments/tabcond/lv/StochasticLV_ABCsamples2k.pk'
    theta1 = np.array([0.01, 0.5, 1, 0.01])
    experiment(StochLV, path_theta1, theta1, flow_picnn)

    path_theta2 = '.../PCPM/experiments/tabcond/lv/StochasticLV_ABCsamplesNewTheta.pk'
    theta2 = np.array([0.02, 0.02, 0.02, 0.02])
    experiment(StochLV, path_theta2, theta2, flow_picnn)
