import argparse
import os
import numpy as np
import datetime
import pandas as pd
import torch
import scipy.io
from torch import distributions
from lib.dataloader import dataloader
from datasets.shallow_water import load_swdata
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.icnn import PICNN
from src.pcpmap import PCPMap
from lib.utils import makedirs, get_logger, AverageMeter

"""
argument parser for hyper parameters and model handling
"""

parser = argparse.ArgumentParser('PCP-Map')
parser.add_argument(
    '--data', choices=['concrete', 'energy', 'yacht', 'lv', 'sw'], type=str, default='concrete'
)
parser.add_argument('--input_x_dim',    type=int, default=1, help="input data convex dimension")
parser.add_argument('--input_y_dim',    type=int, default=8, help="input data non-convex dimension")
parser.add_argument('--out_dim',        type=int, default=1, help="output dimension")

parser.add_argument('--clip',           type=bool, default=True, help="whether clipping the weights or not")
parser.add_argument('--tol',            type=float, default=1e-12, help="LBFGS tolerance")

parser.add_argument('--test_ratio',     type=float, default=0.10, help="test set ratio")
parser.add_argument('--valid_ratio',    type=float, default=0.10, help="validation set ratio")
parser.add_argument('--random_state',   type=int, default=42, help="random state for splitting dataset")
parser.add_argument('--num_epochs',     type=int, default=15, help="pilot run number of epochs")
parser.add_argument('--num_trials',     type=int, default=100, help="pilot run number of trials")

parser.add_argument('--theta_pca',      type=int, default=0, help="project theta in for shallow water")
parser.add_argument('--save',           type=str, default='experiments/tabcond', help="define the save directory")

args = parser.parse_args()

sStartTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# logger
makedirs(args.save)
logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__), saving=True)
logger.info("start time: " + sStartTime)
logger.info(args)

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(data, test_ratio, valid_ratio, batch_size, random_state):
    if data == 'lv':
        # TODO change to correct path
        dataset_load = scipy.io.loadmat('.../PCP-Map/datasets/lv_data.mat')
        x_train = dataset_load['x_train']
        y_train = dataset_load['y_train']
        dataset = np.concatenate((x_train, y_train), axis=1)
        # log transformation over theta
        dataset[:, :4] = np.log(dataset[:, :4])
        # split data and convert to tensor
        train, valid = train_test_split(
            dataset, test_size=valid_ratio,
            random_state=random_state
        )
        train_sz = train.shape[0]

        train_mean = np.mean(train, axis=0, keepdims=True)
        train_std = np.std(train, axis=0, keepdims=True)
        train_data = (train - train_mean) / train_std
        valid_data = (valid - train_mean) / train_std

        # convert to tensor
        train_data = torch.tensor(train_data, dtype=torch.float32)
        valid_data = torch.tensor(valid_data, dtype=torch.float32)

        # load train data
        trn_loader = DataLoader(
            train_data,
            batch_size=batch_size, shuffle=True
        )
        vld_loader = DataLoader(
            valid_data,
            batch_size=batch_size, shuffle=True
        )
    else:
        trn_loader, vld_loader, _, train_sz = dataloader(data, batch_size, test_ratio, valid_ratio, random_state)

    return trn_loader, vld_loader, train_sz


if __name__ == '__main__':

    columns_params = ["batchsz", "lr", "width", "width_y", "depth"]
    columns_valid = ["picnn_nll"]
    params_hist = pd.DataFrame(columns=columns_params)
    valid_hist = pd.DataFrame(columns=columns_valid)

    log_msg = ('{:5s}  {:9s}'.format('trial', 'val_loss'))
    logger.info(log_msg)

    # sample space for hyperparameters
    width_list = np.array([32, 64, 128, 256, 512])
    depth_list = np.array([2, 3, 4, 5, 6])
    if args.data == 'lv' or args.data == 'sw':
        batch_size_list = np.array([64, 128, 256])
    else:
        batch_size_list = np.array([32, 64])
    if args.data == 'sw':
        lr_list = np.array([0.0001])
        if bool(args.theta_pca) is True:
            lr_list = np.array([0.01, 0.001, 0.0001])
    else:
        lr_list = np.array([0.01, 0.005, 0.001])

    for trial in range(args.num_trials):

        batch_size = int(np.random.choice(batch_size_list))
        if args.data == 'sw':
            _, train_loader, valid_data, _ = load_swdata(batch_size)
            if bool(args.theta_pca) is True:
                x_full = train_loader.dataset[:, :args.input_x_dim]
                x_full = x_full.view(-1, args.input_x_dim)
                cov_x = x_full.T @ x_full
                L, V = torch.linalg.eigh(cov_x)
                # get the last dx columns in V
                Vx = V[:, -14:].to(device)
        else:
            train_loader, valid_loader, _ = load_data(args.data, args.test_ratio, args.valid_ratio,
                                                      batch_size, args.random_state)

        if args.clip is True:
            reparam = False
        else:
            reparam = True

        width = np.random.choice(width_list)
        if args.data == "lv" or args.data == "sw":
            width_y = width
        else:
            width_y_list = [width, args.input_y_dim]
            feat_dim = width
            while feat_dim // 2 > args.input_y_dim:
                feat_dim = feat_dim // 2
                width_y_list.append(feat_dim)
            width_y = np.random.choice(width_y_list)
        num_layers = np.random.choice(depth_list)
        lr = np.random.choice(lr_list)

        # Multivariate Gaussian as Reference
        input_x_dim = args.input_x_dim
        if bool(args.theta_pca) is True:
            input_x_dim = 14
        prior_picnn = distributions.MultivariateNormal(torch.zeros(input_x_dim).to(device),
                                                       torch.eye(input_x_dim).to(device))
        # build PCP-Map
        picnn = PICNN(input_x_dim, args.input_y_dim, width, width_y, args.out_dim, num_layers, reparam=reparam)
        pcpmap = PCPMap(prior_picnn, picnn).to(device)
        optimizer = torch.optim.Adam(pcpmap.parameters(), lr=lr)

        params_hist.loc[len(params_hist.index)] = [batch_size, lr, width, width_y, num_layers]

        if args.data == 'lv':
            num_epochs = 2
        elif args.data == 'sw':
            num_epochs = 1
        else:
            num_epochs = args.num_epochs

        for epoch in range(num_epochs):
            for sample in train_loader:
                if args.data == 'lv' or args.data == 'sw':
                    x = sample[:, :args.input_x_dim].requires_grad_(True).to(device)
                    if bool(args.theta_pca) is True:
                        x = x @ Vx
                    y = sample[:, args.input_x_dim:].requires_grad_(True).to(device)
                else:
                    x = sample[:, args.input_y_dim:].requires_grad_(True).to(device)
                    y = sample[:, :args.input_y_dim].requires_grad_(True).to(device)

                # optimizer step for PCP-Map
                optimizer.zero_grad()
                loss = -pcpmap.loglik_picnn(x, y).mean()
                loss.backward()
                optimizer.step()

                # non-negative constraint
                if args.clip is True:
                    for lw in pcpmap.picnn.Lw:
                        with torch.no_grad():
                            lw.weight.data = pcpmap.picnn.nonneg(lw.weight)

        valLossMeterPICNN = AverageMeter()

        if args.data == 'sw':
            x_valid = valid_data[:, :args.input_x_dim].requires_grad_(True).to(device)
            if bool(args.theta_pca) is True:
                x_valid = x_valid @ Vx
            y_valid = valid_data[:, args.input_x_dim:].requires_grad_(True).to(device)
            val_loss_picnn = -pcpmap.loglik_picnn(x_valid, y_valid).mean()
            val_loss_picnn = val_loss_picnn.cpu().detach().numpy()
        else:
            for valid_sample in valid_loader:
                if args.data == 'lv':
                    x_valid = valid_sample[:, :args.input_x_dim].requires_grad_(True).to(device)
                    y_valid = valid_sample[:, args.input_x_dim:].requires_grad_(True).to(device)
                else:
                    x_valid = valid_sample[:, args.input_y_dim:].requires_grad_(True).to(device)
                    y_valid = valid_sample[:, :args.input_y_dim].requires_grad_(True).to(device)
                mean_valid_loss_picnn = -pcpmap.loglik_picnn(x_valid, y_valid).mean()
                valLossMeterPICNN.update(mean_valid_loss_picnn.item(), valid_sample.shape[0])
            val_loss_picnn = valLossMeterPICNN.avg

        log_message = '{:05d}  {:9.3e}'.format(trial+1, val_loss_picnn)
        logger.info(log_message)
        valid_hist.loc[len(valid_hist.index)] = [val_loss_picnn]

    params_hist.to_csv(os.path.join(args.save, '%s_params_hist.csv' % args.data))
    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % args.data))
