import argparse
import os
import numpy as np
import datetime
import pandas as pd
import torch
from torch import distributions
from lib.dataloader import dataloader
from src.icnn import FICNN, PICNN
from src.triflow_ficnn import TriFlowFICNN
from src.triflow_picnn import TriFlowPICNN
from lib.utils import makedirs, get_logger, AverageMeter

"""
argument parser for hyper parameters and model handling
"""

parser = argparse.ArgumentParser('PCPM')
parser.add_argument(
    '--data', choices=['wt_wine', 'rd_wine', 'parkinson'], type=str, default='rd_wine'
)
parser.add_argument('--input_x_dim',    type=int, default=6, help="input data convex dimension")
parser.add_argument('--input_y_dim',    type=int, default=5, help="input data non-convex dimension")
parser.add_argument('--out_dim',        type=int, default=1, help="output dimension")

parser.add_argument('--clip',           type=bool, default=True, help="whether clipping the weights or not")
parser.add_argument('--tol',            type=float, default=1e-12, help="LBFGS tolerance")

parser.add_argument('--test_ratio',     type=float, default=0.10, help="test set ratio")
parser.add_argument('--valid_ratio',    type=float, default=0.10, help="validation set ratio")
parser.add_argument('--random_state',   type=int, default=42, help="random state for splitting dataset")
parser.add_argument('--num_epochs',     type=int, default=4, help="pilot run number of epochs")

parser.add_argument('--save',           type=str, default='experiments/tabjoint', help="define the save directory")

args = parser.parse_args()

sStartTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# logger
makedirs(args.save)
logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__), saving=True)
logger.info("start time: " + sStartTime)
logger.info(args)

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    columns_params = ["batchsz", "lr", "width", "depth"]
    columns_valid = ["ficnn_nll", "picnn_nll", "tot_nll"]
    params_hist = pd.DataFrame(columns=columns_params)
    valid_hist = pd.DataFrame(columns=columns_valid)

    log_msg = ('{:5s}  {:9s}  {:9s}  {:9s}'.format('trial', 'val_ficnn', 'val_picnn', 'val_loss'))
    logger.info(log_msg)

    # sample space for hyperparameters
    width_list = np.array([32, 64, 128, 256, 512])
    depth_list = np.array([2, 3, 4, 5, 6])
    batch_size_list = np.array([32, 64])
    lr_list = np.array([0.01, 0.005, 0.001])

    for trial in range(80):

        batch_size = int(np.random.choice(batch_size_list))
        train_loader, valid_loader, _, train_size = dataloader(args.data, batch_size, args.test_ratio,
                                                               args.valid_ratio, args.random_state)

        # Establishing TC-Flows
        if args.clip is True:
            reparam = False
        else:
            reparam = True

        width = np.random.choice(width_list)
        width_y_list = [width]
        feat_dim = width
        while feat_dim >= args.input_y_dim:
            feat_dim = feat_dim // 2
            width_y_list.append(feat_dim)
        width_y = np.random.choice(width_y_list)
        num_layers = np.random.choice(depth_list)
        lr = np.random.choice(lr_list)

        # Multivariate Gaussian as Reference
        prior_ficnn = distributions.MultivariateNormal(torch.zeros(args.input_y_dim).to(device),
                                                       torch.eye(args.input_y_dim).to(device))
        prior_picnn = distributions.MultivariateNormal(torch.zeros(args.input_x_dim).to(device),
                                                       torch.eye(args.input_x_dim).to(device))

        # establish TC-Flow
        ficnn = FICNN(args.input_y_dim, width, args.out_dim, num_layers, reparam=reparam).to(device)
        picnn = PICNN(args.input_x_dim, args.input_y_dim, width, width_y, args.out_dim, num_layers, reparam=reparam).to(device)

        flow_ficnn = TriFlowFICNN(prior_ficnn, ficnn).to(device)
        flow_picnn = TriFlowPICNN(prior_picnn, picnn).to(device)

        optimizer1 = torch.optim.Adam(flow_ficnn.parameters(), lr=lr)
        optimizer2 = torch.optim.Adam(flow_picnn.parameters(), lr=lr)

        params_hist.loc[len(params_hist.index)] = [batch_size, lr, width, width_y, num_layers]

        if args.data == 'parkinson' or args.data == 'wt_wine':
            num_epochs = args.num_epochs
        else:
            num_epochs = 8

        for epoch in range(num_epochs):
            for sample in train_loader:
                x = sample[:, args.input_y_dim:].requires_grad_(True).to(device)
                y = sample[:, :args.input_y_dim].requires_grad_(True).to(device)

                # optimizer step for flow1
                optimizer1.zero_grad()
                loss1 = -flow_ficnn.loglik_ficnn(y).mean()
                loss1.backward()
                optimizer1.step()

                # non-negative constraint
                if args.clip is True:
                    for lz in flow_ficnn.ficnn.Lz:
                        with torch.no_grad():
                            lz.weight.data = flow_ficnn.ficnn.nonneg(lz.weight)

                # optimizer step for flow2
                optimizer2.zero_grad()
                loss2 = -flow_picnn.loglik_picnn(x, y).mean()
                loss2.backward()
                optimizer2.step()

                # non-negative constraint
                if args.clip is True:
                    for lw in flow_picnn.picnn.Lw:
                        with torch.no_grad():
                            lw.weight.data = flow_picnn.picnn.nonneg(lw.weight)

        valLossMeterFICNN = AverageMeter()
        valLossMeterPICNN = AverageMeter()

        for valid_sample in valid_loader:
            x_valid = valid_sample[:, args.input_y_dim:].requires_grad_(True).to(device)
            y_valid = valid_sample[:, :args.input_y_dim].requires_grad_(True).to(device)
            mean_valid_loss_ficnn = -flow_ficnn.loglik_ficnn(y_valid).mean()
            mean_valid_loss_picnn = -flow_picnn.loglik_picnn(x_valid, y_valid).mean()
            valLossMeterFICNN.update(mean_valid_loss_ficnn.item(), valid_sample.shape[0])
            valLossMeterPICNN.update(mean_valid_loss_picnn.item(), valid_sample.shape[0])

        val_loss_ficnn = valLossMeterFICNN.avg
        val_loss_picnn = valLossMeterPICNN.avg
        val_loss = val_loss_ficnn + val_loss_picnn

        log_message = '{:05d}  {:9.3e}  {:9.3e}  {:9.3e}  '.format(trial+1, val_loss_ficnn, val_loss_picnn, val_loss)
        logger.info(log_message)
        valid_hist.loc[len(valid_hist.index)] = [val_loss_ficnn, val_loss_picnn, val_loss]

    params_hist.to_csv(os.path.join(args.save, '%s_params_hist.csv' % args.data))
    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % args.data))
