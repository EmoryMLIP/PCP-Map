import argparse
import scipy.io
import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch import distributions
from src.icnn import PICNN
from src.plotter import plot_matrix
from src.pcpmap import PCPMap
from datasets.StochasticLotkaVolterra import StochasticLotkaVolterra
import matplotlib.pyplot as plt
from lib.utils import AverageMeter

parser = argparse.ArgumentParser('PCP-Map')
parser.add_argument('--resume',    type=str, default="/experiments/tabcond/lv/...")

args = parser.parse_args()


def experiment(LV, abc_dat_path, theta_star, model, trn_mean, trn_std, checkpt):

    """grab y star from ABC"""
    input_x_dim = checkpt['args'].input_x_dim
    abc_sample = pd.read_pickle(abc_dat_path)
    y_theta_star = abc_sample["y_true"]
    y_theta_star_norm = (y_theta_star - trn_mean[:, input_x_dim:]) / trn_std[:, input_x_dim:]
    y_theta_star_norm_tensor = torch.tensor(y_theta_star_norm, dtype=torch.float32)

    """generate samples"""

    zx = torch.randn(2000, 4).to(device)
    # start sampling timer
    start = time.time()
    x_gen, num_evals = model.gx(zx, y_theta_star_norm_tensor.to(device), checkpt['args'].tol)
    # end timer
    sample_time = time.time() - start
    print(f"Sampling Time for theta {theta_star[0].item()}, tol={checkpt['args'].tol}: {sample_time}")
    print(f"Number of closure calls for theta {theta_star[0].item()}, tol={checkpt['args'].tol}: {num_evals}")
    print("Number of closure calls: " + str(num_evals))
    theta_gen = x_gen.detach().cpu().numpy()
    theta_gen = (theta_gen * trn_std[:, :input_x_dim] + trn_mean[:, :input_x_dim]).squeeze()

    """tolerance decrement experiment"""

    tol_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    for tol in tol_list:
        # start sampling timer
        start_tol = time.time()
        x_gen_tol, num_evals_tol = model.gx(zx, y_theta_star_norm_tensor.to(device), tol)
        # end timer
        sample_time_tol = time.time() - start_tol
        print("Sampling Time for tol=" + str(tol) + " is: " + str(sample_time_tol))
        print("Number of closure calls for tol= " + str(tol) + " is: " + str(num_evals_tol))
        x_gen_tol = x_gen_tol.detach().to(device)
        theta_gen_tol = x_gen_tol.detach().cpu().numpy()
        theta_gen_tol = (theta_gen_tol * trn_std[:, :input_x_dim] + trn_mean[:, :input_x_dim]).squeeze()
        # calculate normed error
        error = np.linalg.norm(theta_gen - theta_gen_tol) / np.linalg.norm(theta_gen)
        print(f"Norm Error for theta {theta_star[0].item()} tol={tol}: {error}")

    """plot"""

    theta_star_log = np.log(theta_star)
    symbols = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$']
    log_limits = [[-5., 2.], [-5., 2.], [-5., 2.], [-5., 2.]]
    plot_matrix(theta_gen, log_limits, xtrue=theta_star_log, symbols=symbols)
    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_' + str(theta_star[0].item()) + '.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    # plot for ABC
    theta_abc = np.array(abc_sample['all_x'])[-1]
    theta_abc = theta_abc.reshape(-1, 4)
    theta_abc = np.log(theta_abc)
    plot_matrix(theta_abc, log_limits, xtrue=theta_star_log, symbols=symbols)
    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_' + str(theta_star[0].item()) + '_abc.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

    """plot posterior predictive"""

    plt.figure()
    ytrue = LV.simulate(theta_star)
    c1 = plt.plot(LV.tt, ytrue[:, 0], '-', label='Predators')
    c2 = plt.plot(LV.tt, ytrue[:, 1], '-', label='Prey')
    for _ in range(10):
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

    # plot ABC posterior predictive
    plt.figure()
    ytrue = LV.simulate(theta_star)
    c1 = plt.plot(LV.tt, ytrue[:, 0], '-', label='Predators')
    c2 = plt.plot(LV.tt, ytrue[:, 1], '-', label='Prey')
    for _ in range(10):
        rand_sample = np.random.randint(low=0, high=2000, size=(1,))[0]
        xi = np.exp(theta_abc[rand_sample, :])
        yt = LV.simulate(xi)
        plt.plot(LV.tt, yt[:, 0], '--', color=c1[0].get_color(), alpha=0.3)
        plt.plot(LV.tt, yt[:, 1], '--', color=c2[0].get_color(), alpha=0.3)
    plt.xlabel('$t$', size=20)
    plt.ylabel('$Z(t)$', size=20)
    plt.legend(loc='upper right')
    plt.xlim(0, 20)
    sPath = os.path.join(checkpt['args'].save, 'figs',
                         checkpt['args'].data + '_' + str(theta_star[0].item()) + '_abc_post.png')
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()


if __name__ == '__main__':

    """Load Saved Model"""

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

    """Testing"""

    dataset_load = scipy.io.loadmat('.../PCP-Map/datasets/training_data.mat')
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

    # TODO change to correct paths
    StochLV = StochasticLotkaVolterra()
    path_theta1 = '.../PCP-Map/experiments/tabcond/lv/StochasticLV_ABCsamples01.pk'
    theta1 = np.array([0.01, 0.5, 1, 0.01])
    experiment(StochLV, path_theta1, theta1, pcpmap, train_mean, train_std, checkpt)

    path_theta2 = '.../PCP-Map/experiments/tabcond/lv/StochasticLV_ABCsamples015NewTheta.pk'
    theta2 = np.array([0.02, 0.02, 0.02, 0.02])
    experiment(StochLV, path_theta2, theta2, pcpmap, train_mean, train_std, checkpt)

    """Density Estimation"""
    # TODO change to correct path
    test_dataset_load = scipy.io.loadmat('.../PCPM/datasets/lv_test_data.mat')
    test_dat = test_dataset_load['test_data']
    # log transformation over theta
    test_dat[:, :4] = np.log(test_dat[:, :4])
    test_data = (test_dat - train_mean) / train_std
    test_data = torch.tensor(test_data, dtype=torch.float32)
    tst_loader = DataLoader(test_data, batch_size=128, shuffle=True)
    tstLossMeter = AverageMeter()
    tsttimeMeter = AverageMeter()
    for xy in tst_loader:
        x_test = xy[:, :input_x_dim].requires_grad_(True).to(device)
        y_test = xy[:, input_x_dim:].requires_grad_(True).to(device)
        # start timer
        end_tst = time.time()
        test_loss = -pcpmap.loglik_picnn(x_test, y_test).mean()
        # end timer
        tststep_time = time.time() - end_tst
        tsttimeMeter.update(tststep_time)
        tstLossMeter.update(test_loss.item(), xy.shape[0])
    print("Test NLL: " + str(tstLossMeter.avg))
    print("Test time: " + str(tsttimeMeter.sum))
