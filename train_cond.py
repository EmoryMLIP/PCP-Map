import argparse
import os
import time
import datetime
import scipy.io
import numpy as np
import pandas as pd
import torch
from torch import distributions
from lib.dataloader import dataloader
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.icnn import PICNN
from src.pcpmap import PCPMap
from src.mmd import mmd
from datasets.shallow_water import load_swdata
from lib.utils import count_parameters, makedirs, get_logger, AverageMeter

"""
argument parser for hyper parameters and model handling
"""

parser = argparse.ArgumentParser('PCP-Map')
parser.add_argument(
    '--data', choices=['concrete', 'energy', 'yacht', 'lv', 'sw'], type=str, default='lv'
)
parser.add_argument('--input_x_dim',    type=int, default=4, help="input data convex dimension")
parser.add_argument('--input_y_dim',    type=int, default=9, help="input data non-convex dimension")
parser.add_argument('--feature_dim',    type=int, default=128, help="intermediate layer feature dimension")
parser.add_argument('--feature_y_dim',  type=int, default=128, help="intermediate layer context dimension")
parser.add_argument('--out_dim',        type=int, default=1, help="output dimension")
parser.add_argument('--num_layers_pi',  type=int, default=2, help="depth of PICNN network")

parser.add_argument('--clip',           type=bool, default=True, help="whether clipping the weights or not")
parser.add_argument('--tol',            type=float, default=1e-6, help="LBFGS tolerance")

parser.add_argument('--batch_size',     type=int, default=256, help="number of samples per batch")
parser.add_argument('--num_epochs',     type=int, default=1000, help="number of training steps")
parser.add_argument('--print_freq',     type=int, default=1, help="how often to print results to log")
parser.add_argument('--valid_freq',     type=int, default=50, help="how often to run model on validation set")
parser.add_argument('--early_stopping', type=int, default=20, help="early stopping of training based on validation")
parser.add_argument('--lr',             type=float, default=0.005, help="optimizer learning rate")
parser.add_argument("--lr_drop",        type=float, default=2.0, help="how much to decrease lr (divide by)")

parser.add_argument('--test_ratio',     type=float, default=0.10, help="test set ratio")
parser.add_argument('--valid_ratio',    type=float, default=0.10, help="validation set ratio")
parser.add_argument('--random_state',   type=int, default=42, help="random state for splitting dataset")

parser.add_argument('--save_test',      type=int, default=1, help="if 1 then saves test numerics 0 if not")
parser.add_argument('--save',           type=str, default='experiments/cond', help="define the save directory")
parser.add_argument('--theta_pca',      type=int, default=0, help="project theta in for shallow water")

args = parser.parse_args()

sStartTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# logger
makedirs(args.save)
logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__), saving=True)
logger.info("start time: " + sStartTime)
logger.info(args)

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# decrease the learning rate based on validation
ndecs_picnn = 0
n_vals_wo_improve_picnn = 0
def update_lr_picnn(optimizer, n_vals_without_improvement):
    global ndecs_picnn
    if ndecs_picnn == 0 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop
        ndecs_picnn = 1
    elif ndecs_picnn == 1 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop**2
        ndecs_picnn = 2
    else:
        ndecs_picnn += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop**ndecs_picnn


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


def evaluate_model(model, data, batch_size, test_ratio, valid_ratio, random_state, input_y_dim, input_x_dim, tol,
                   bestParams_picnn):

    _, _, testData, _ = dataloader(data, batch_size, test_ratio, valid_ratio, random_state)

    # Load Best Models
    model.load_state_dict(bestParams_picnn)
    model = model.to(device)
    # Obtain test metrics numbers
    x_test = testData[:, input_y_dim:].requires_grad_(True).to(device)
    y_test = testData[:, :input_y_dim].requires_grad_(True).to(device)
    log_prob_picnn = model.loglik_picnn(x_test, y_test)
    pb_mean_NLL = -log_prob_picnn.mean()
    # Calculate MMD
    zx = torch.randn(testData.shape[0], input_x_dim).to(device)
    x_generated, _ = model.gx(zx, testData[:, :input_y_dim].to(device), tol=tol)
    x_generated = x_generated.detach().to(device)
    mean_max_dis = mmd(x_generated, testData[:, input_y_dim:].to(device))

    return pb_mean_NLL.item(), mean_max_dis.item()


"""
Training Process
"""

if __name__ == '__main__':

    """Load Data"""

    if args.data == 'sw':
        _, train_loader, valid_data, n_train = load_swdata(args.batch_size)
        if bool(args.theta_pca) is True:
            x_full = train_loader.dataset[:, :args.input_x_dim]
            x_full = x_full.view(-1, args.input_x_dim)
            cov_x = x_full.T @ x_full
            L, V = torch.linalg.eigh(cov_x)
            # get the last dx columns in V
            Vx = V[:, -14:].to(device)
    else:
        train_loader, valid_loader, n_train = load_data(args.data, args.test_ratio, args.valid_ratio,
                                                        args.batch_size, args.random_state)

    """Construct Model"""

    if args.clip is True:
        reparam = False
    else:
        reparam = True

    # Multivariate Gaussian as Reference
    input_x_dim = args.input_x_dim
    if args.data == 'sw' and bool(args.theta_pca) is True:
        input_x_dim = 14
    prior_picnn = distributions.MultivariateNormal(torch.zeros(input_x_dim).to(device), torch.eye(input_x_dim).to(device))
    # build PCP-Map
    picnn = PICNN(input_x_dim, args.input_y_dim, args.feature_dim, args.feature_y_dim,
                  args.out_dim, args.num_layers_pi, reparam=reparam)
    pcpmap = PCPMap(prior_picnn, picnn).to(device)

    optimizer = torch.optim.Adam(pcpmap.parameters(), lr=args.lr)

    """Initial Logs"""

    strTitle = args.data + '_' + sStartTime + '_' + str(args.batch_size) + '_' + str(args.lr) + \
               '_' + str(args.num_layers_pi) + '_' + str(args.feature_dim)

    logger.info("--------------------------------------------------")
    logger.info("Number of trainable parameters: {}".format(count_parameters(picnn)))
    logger.info("--------------------------------------------------")
    logger.info(str(optimizer))  # optimizer info
    logger.info("--------------------------------------------------")
    logger.info("device={:}".format(device))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info("--------------------------------------------------\n")

    columns_train = ["epoch", "step", "time/trnstep", "train_loss_p"]
    columns_valid = ["time/vldstep", "valid_loss_p"]
    train_hist = pd.DataFrame(columns=columns_train)
    valid_hist = pd.DataFrame(columns=columns_valid)

    logger.info(["iter"] + columns_train)

    """Training Starts"""

    # starts training
    itr = 1
    total_itr = (int(n_train / args.batch_size) + 1) * args.num_epochs
    best_loss_picnn = float('inf')
    bestParams_picnn = None

    makedirs(args.save)
    timeMeter = AverageMeter()
    vldTotTimeMeter = AverageMeter()

    for epoch in range(args.num_epochs):
        for i, sample in enumerate(train_loader):
            if args.data == 'lv' or args.data == 'sw':
                x = sample[:, :args.input_x_dim].requires_grad_(True).to(device)
                if bool(args.theta_pca) is True:
                    x = x @ Vx
                y = sample[:, args.input_x_dim:].requires_grad_(True).to(device)
            else:
                x = sample[:, args.input_y_dim:].requires_grad_(True).to(device)
                y = sample[:, :args.input_y_dim].requires_grad_(True).to(device)

            # start timer
            end = time.time()

            optimizer.zero_grad()
            loss = -pcpmap.loglik_picnn(x, y).mean()
            loss.backward()
            optimizer.step()

            # non-negative constraint
            if args.clip is True:
                for lw in pcpmap.picnn.Lw:
                    with torch.no_grad():
                        lw.weight.data = pcpmap.picnn.nonneg(lw.weight)

            # end timer
            step_time = time.time() - end
            timeMeter.update(step_time)
            train_hist.loc[len(train_hist.index)] = [epoch + 1, i + 1, step_time, loss.item()]

            # printing
            if itr % args.print_freq == 0:
                log_message = (
                    '{:05d}  {:7.1f}     {:04d}    {:9.3e}      {:9.3e} '.format(
                        itr, epoch + 1, i + 1, step_time, loss.item()
                    )
                )
                logger.info(log_message)

            if itr % args.valid_freq == 0 or itr == total_itr:

                if args.data == 'sw':
                    x_valid = valid_data[:, :args.input_x_dim].requires_grad_(True).to(device)
                    if bool(args.theta_pca) is True:
                        x_valid = x_valid @ Vx
                    y_valid = valid_data[:, args.input_x_dim:].requires_grad_(True).to(device)
                    # start timer
                    end_vld = time.time()
                    val_loss_picnn = -pcpmap.loglik_picnn(x_valid, y_valid).mean()
                    # end timer
                    vldstep_time = time.time() - end_vld
                    vldTotTimeMeter.update(vldstep_time)
                    val_loss_picnn = val_loss_picnn.cpu().detach().numpy()
                else:
                    vldtimeMeter = AverageMeter()
                    valLossMeterPICNN = AverageMeter()
                    for valid_sample in valid_loader:
                        if args.data == 'lv':
                            x_valid = valid_sample[:, :args.input_x_dim].requires_grad_(True).to(device)
                            y_valid = valid_sample[:, args.input_x_dim:].requires_grad_(True).to(device)
                        else:
                            x_valid = valid_sample[:, args.input_y_dim:].requires_grad_(True).to(device)
                            y_valid = valid_sample[:, :args.input_y_dim].requires_grad_(True).to(device)
                        # start timer
                        end_vld = time.time()
                        mean_valid_loss_picnn = -pcpmap.loglik_picnn(x_valid, y_valid).mean()
                        # end timer
                        batch_step_time = time.time() - end_vld
                        vldtimeMeter.update(batch_step_time)
                        valLossMeterPICNN.update(mean_valid_loss_picnn.item(), valid_sample.shape[0])
                    val_loss_picnn = valLossMeterPICNN.avg
                    vldstep_time = vldtimeMeter.sum
                    vldTotTimeMeter.update(vldstep_time)

                valid_hist.loc[len(valid_hist.index)] = [vldstep_time, val_loss_picnn]
                log_message_valid = '   {:9.3e}      {:9.3e} '.format(vldstep_time, val_loss_picnn)

                if val_loss_picnn < best_loss_picnn:
                    n_vals_wo_improve_picnn = 0
                    best_loss_picnn = val_loss_picnn
                    makedirs(args.save)
                    bestParams_picnn = pcpmap.state_dict()
                    torch.save({
                        'args': args,
                        'state_dict_picnn': bestParams_picnn,
                    }, os.path.join(args.save, strTitle + '_checkpt.pth'))
                else:
                    n_vals_wo_improve_picnn += 1
                log_message_valid += '    picnn no improve: {:d}/{:d}'.format(n_vals_wo_improve_picnn,
                                                                              args.early_stopping)

                logger.info(columns_valid)
                logger.info(log_message_valid)
                logger.info(["iter"] + columns_train)

            # update learning rate
            if n_vals_wo_improve_picnn > args.early_stopping:
                if ndecs_picnn > 2:
                    logger.info("early stopping engaged")
                    logger.info("Training Time: {:} seconds".format(timeMeter.sum))
                    logger.info("Validation Time: {:} seconds".format(vldTotTimeMeter.sum))
                    train_hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % strTitle))
                    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % strTitle))
                    if bool(args.save_test) is False:
                        exit(0)
                    elif args.data == 'sw':
                        os.system(
                            "python evaluate_sw.py --resume " + ".../PCP-Map" + args.save + "/" + strTitle + '_checkpt.pth'
                        )
                        exit(0)
                    elif args.data == 'lv':
                        os.system(
                            "python evaluate_lv.py --resume " + ".../PCP-Map/" + args.save + "/" + strTitle + '_checkpt.pth'
                        )
                        exit(0)
                    else:
                        NLL, MMD = evaluate_model(pcpmap, args.data, args.batch_size, args.test_ratio, args.valid_ratio,
                                                  args.random_state, args.input_y_dim, args.input_x_dim, args.tol,
                                                  bestParams_picnn)
                        columns_test = ["batch_size", "lr", "width", "width_y", "depth", "NLL", "MMD", "time", "iter"]
                        test_hist = pd.DataFrame(columns=columns_test)
                        test_hist.loc[len(test_hist.index)] = [args.batch_size, args.lr, args.feature_dim,
                                                               args.feature_y_dim,
                                                               args.num_layers_pi, NLL, MMD, timeMeter.sum, itr]
                        testfile_name = '.../PCP-Map/experiments/tabcond/' + args.data + '_test_hist.csv'
                        if os.path.isfile(testfile_name):
                            test_hist.to_csv(testfile_name, mode='a', index=False, header=False)
                        else:
                            test_hist.to_csv(testfile_name, index=False)
                        exit(0)
                else:
                    update_lr_picnn(optimizer, n_vals_wo_improve_picnn)
                    n_vals_wo_improve_picnn = 0

            itr += 1

    print('Training time: %.2f secs' % timeMeter.sum)
    print('Validation time: %.2f secs' % vldTotTimeMeter.sum)
    train_hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % strTitle))
    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % strTitle))
    if bool(args.save_test) is False:
        exit(0)
    elif args.data == 'sw':
        os.system(
            "python evaluate_sw.py --resume " + ".../PCP-Map/" + args.save + "/" + strTitle + '_checkpt.pth'
        )
        exit(0)
    elif args.data == 'lv':
        os.system(
            "python evaluate_lv.py --resume " + ".../PCP-Map/" + args.save + "/" + strTitle + '_checkpt.pth'
        )
        exit(0)
    else:
        NLL, MMD = evaluate_model(pcpmap, args.data, args.batch_size, args.test_ratio, args.valid_ratio,
                                  args.random_state, args.input_y_dim, args.input_x_dim, args.tol, bestParams_picnn)

        columns_test = ["batch_size", "lr", "width", "width_y", "depth", "NLL", "MMD", "time", "iter"]
        test_hist = pd.DataFrame(columns=columns_test)
        test_hist.loc[len(test_hist.index)] = [args.batch_size, args.lr, args.feature_dim, args.feature_y_dim,
                                               args.num_layers_pi, NLL, MMD,
                                               timeMeter.sum, itr]
        testfile_name = '.../PCP-Map/experiments/tabcond/' + args.data + '_test_hist.csv'
        if os.path.isfile(testfile_name):
            test_hist.to_csv(testfile_name, mode='a', index=False, header=False)
        else:
            test_hist.to_csv(testfile_name, index=False)
