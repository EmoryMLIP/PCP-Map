import argparse
import datetime
import os
import pandas as pd
import torch
from torch import distributions
from torch.utils.data import DataLoader
from datasets import toy_data
from src.icnn import FICNN, PICNN
from src.triflow_ficnn import TriFlowFICNN
from src.triflow_picnn import TriFlowPICNN
from lib.utils import count_parameters, makedirs, get_logger
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno'

"""
argument parser for hyper parameters and model handling
"""

parser = argparse.ArgumentParser('PCPM')
parser.add_argument(
    '--data', choices=['moon', 'spiral', 'circles', 'swiss', '2spirals',
                       'checkerboard', 'pinwheel', '8gauss'],
    type=str, default='moon'
)
parser.add_argument('--data_size',      type=int, default=30000, help="input data fully convex dimension")

parser.add_argument('--input_x_dim',    type=int, default=1, help="input data convex dimension")
parser.add_argument('--input_y_dim',    type=int, default=1, help="input data non-convex dimension")
parser.add_argument('--feature_dim',    type=int, default=256, help="intermediate layer feature dimension")
parser.add_argument('--out_dim',        type=int, default=1, help="output dimension")
parser.add_argument('--num_layers_fi',  type=int, default=3, help="depth of FICNN network")
parser.add_argument('--num_layers_pi',  type=int, default=3, help="depth of PICNN network")

parser.add_argument('--swap',           type=bool, default=False, help="whether swapping the dimension or not")
parser.add_argument('--clip',           type=bool, default=True, help="whether clipping the weights or not")
parser.add_argument('--tol',            type=float, default=1e-12, help="LBFGS tolerance")

parser.add_argument('--batch_size',     type=int, default=128, help="number of samples per batch")
parser.add_argument('--num_epochs',     type=int, default=1000, help="number of training steps")
parser.add_argument('--print_freq',     type=int, default=10, help="printing frequency")
parser.add_argument('--valid_freq',     type=int, default=50, help="how often to run model on validation set")
parser.add_argument('--valid_size',     type=int, default=512, help="size of validation set")
parser.add_argument('--early_stopping', type=int, default=20, help="early stopping of training based on validation")
parser.add_argument('--lr',             type=float, default=0.01, help="optimizer learning rate")
parser.add_argument("--lr_drop",        type=float, default=2.0, help="how much to decrease lr (divide by)")
parser.add_argument('--sample_size',    type=int, default=100000, help="sample size for plotting")

parser.add_argument('--save',           type=str, default='experiments/toy', help="define the save directory")

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
ndecs_ficnn = 0
n_vals_wo_improve_ficnn = 0
def update_lr_ficnn(optimizer, n_vals_without_improvement):
    global ndecs_ficnn
    if ndecs_ficnn == 0 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop
        ndecs_ficnn = 1
    elif ndecs_ficnn == 1 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop ** 2
        ndecs_ficnn = 2
    else:
        ndecs_ficnn += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop**ndecs_ficnn


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


if __name__ == '__main__':

    # loading dataset
    dat = args.data
    if dat == "moon":
        dataset = toy_data.get_moon(args.data_size)
        test_set = toy_data.get_moon(args.sample_size)
        valid_set = toy_data.get_moon(args.valid_size)
    elif dat == "spiral":
        dataset = toy_data.get_spiral(args.data_size)
        test_set = toy_data.get_spiral(args.sample_size)
        valid_set = toy_data.get_spiral(args.valid_size)
    elif dat == "circles":
        dataset = toy_data.get_circles(args.data_size)
        test_set = toy_data.get_circles(args.sample_size)
        valid_set = toy_data.get_circles(args.valid_size)
    elif dat == "swiss":
        dataset = toy_data.get_swiss_roll(args.data_size)
        test_set = toy_data.get_swiss_roll(args.sample_size)
        valid_set = toy_data.get_swiss_roll(args.valid_size)
    elif dat == "pinwheel":
        dataset = toy_data.get_pinwheel(args.data_size)
        test_set = toy_data.get_pinwheel(args.sample_size)
        valid_set = toy_data.get_pinwheel(args.valid_size)
    elif dat == "8gauss":
        dataset = toy_data.get_8gauss(args.data_size)
        test_set = toy_data.get_8gauss(args.sample_size)
        valid_set = toy_data.get_8gauss(args.valid_size)
    elif dat == "2spirals":
        dataset = toy_data.get_2spirals(args.data_size)
        test_set = toy_data.get_2spirals(args.sample_size)
        valid_set = toy_data.get_2spirals(args.valid_size)
    elif dat == "checkerboard":
        dataset = toy_data.get_checkerboard(args.data_size)
        test_set = toy_data.get_checkerboard(args.sample_size)
        valid_set = toy_data.get_checkerboard(args.valid_size)
    else:
        raise Exception("Dataset is Incorrect")

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    """
    Establishing Triangular Flows
    """

    if args.clip is True:
        reparam = False
    else:
        reparam = True

    prior = distributions.MultivariateNormal(torch.zeros(1).to(device), torch.eye(1).to(device))
    ficnn = FICNN(args.input_y_dim, args.feature_dim, args.out_dim, args.num_layers_fi, reparam=reparam).to(device)
    picnn = PICNN(args.input_x_dim, args.input_y_dim, args.feature_dim, args.out_dim, args.num_layers_pi, reparam=reparam).to(device)

    flow_ficnn = TriFlowFICNN(prior, ficnn).to(device)
    flow_picnn = TriFlowPICNN(prior, picnn).to(device)

    optimizer1 = torch.optim.Adam(flow_ficnn.parameters(), lr=args.lr)
    optimizer2 = torch.optim.Adam(flow_picnn.parameters(), lr=args.lr)

    strTitle = args.data + '_' + sStartTime

    logger.info("--------------------------------------------------")
    logger.info("Number of trainable parameters: {}".format(count_parameters(ficnn) + count_parameters(picnn)))
    logger.info("--------------------------------------------------")
    logger.info(str(optimizer1))  # optimizer info
    logger.info(str(optimizer2))
    logger.info("--------------------------------------------------")
    logger.info("device={:}".format(device))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info("--------------------------------------------------\n")

    columns_train = ["epoch", "step", "train_loss_f", "train_loss_p"]
    columns_valid = ["valid_loss_f", "valid_loss_p"]
    train_hist = pd.DataFrame(columns=columns_train)
    valid_hist = pd.DataFrame(columns=columns_valid)

    logger.info(["iter"] + columns_train)

    """
    Training
    """

    itr = 1
    total_itr = (int(args.data_size / args.batch_size) + 1) * args.num_epochs
    best_loss_ficnn = float('inf')
    best_loss_picnn = float('inf')
    bestParams_ficnn = None
    bestParams_picnn = None

    makedirs(args.save)

    for epoch in range(args.num_epochs):
        for i, sample in enumerate(train_loader):
            x = sample[:, 1].view(-1, 1).requires_grad_(True).to(device)
            y = sample[:, 0].view(-1, 1).requires_grad_(True).to(device)

            # optimizer step for flow1
            optimizer1.zero_grad()
            if args.swap is False:
                loss1 = -flow_ficnn.loglik_ficnn(y).mean()
            else:
                loss1 = -flow_ficnn.loglik_ficnn(x).mean()
            loss1.backward()
            optimizer1.step()

            # non-negative constraint
            if args.clip is True:
                with torch.no_grad():
                    for lz in flow_ficnn.ficnn.Lz:
                        lz.weight.data = flow_ficnn.ficnn.nonneg(lz.weight)

            # optimizer step for flow2
            optimizer2.zero_grad()
            if args.swap is False:
                loss2 = -flow_picnn.loglik_picnn(x, y).mean()
            else:
                loss2 = -flow_picnn.loglik_picnn(y, x).mean()
            loss2.backward()
            optimizer2.step()

            # non-negative constraint
            if args.clip is True:
                for lw in flow_picnn.picnn.Lw:
                    with torch.no_grad():
                        lw.weight.data = flow_picnn.picnn.nonneg(lw.weight)

            train_hist.loc[len(train_hist.index)] = [epoch + 1, i + 1, loss1.item(), loss2.item()]

            # printing
            if itr % args.print_freq == 0:
                log_message = (
                    '{:05d}  {:7.1f}     {:04d}    {:9.3e}       {:9.3e} '.format(
                        itr, epoch + 1, i + 1, loss1.item(), loss2.item()
                    )
                )
                logger.info(log_message)

            if itr % args.valid_freq == 0 or itr % total_itr == 0:
                x_valid = valid_set[:, 1].view(-1, 1).requires_grad_(True).to(device)
                y_valid = valid_set[:, 0].view(-1, 1).requires_grad_(True).to(device)

                if args.swap is False:
                    mean_valid_loss_ficnn = -flow_ficnn.loglik_ficnn(y_valid).mean()
                    mean_valid_loss_picnn = -flow_picnn.loglik_picnn(x_valid, y_valid).mean()
                else:
                    mean_valid_loss_ficnn = -flow_ficnn.loglik_ficnn(x_valid).mean()
                    mean_valid_loss_picnn = -flow_picnn.loglik_picnn(y_valid, x_valid).mean()

                valid_hist.loc[len(valid_hist.index)] = [mean_valid_loss_ficnn.item(), mean_valid_loss_picnn.item()]

                log_message_valid = '   {:9.3e}       {:9.3e} '.format(
                    mean_valid_loss_ficnn.item(),  mean_valid_loss_picnn.item()
                )

                if mean_valid_loss_ficnn.item() < best_loss_ficnn:
                    n_vals_wo_improve_ficnn = 0
                    best_loss_ficnn = mean_valid_loss_ficnn.item()
                    makedirs(args.save)
                    bestParams_ficnn = flow_ficnn.state_dict()
                    torch.save({
                        'args': args,
                        'state_dict_ficnn': bestParams_ficnn,
                        'state_dict_picnn': bestParams_picnn,
                    }, os.path.join(args.save, strTitle + '_checkpt.pth'))
                else:
                    n_vals_wo_improve_ficnn += 1
                log_message_valid += '   ficnn no improve: {:d}/{:d}  '.format(n_vals_wo_improve_ficnn,
                                                                               args.early_stopping)

                if mean_valid_loss_picnn.item() < best_loss_picnn:
                    n_vals_wo_improve_picnn = 0
                    best_loss_picnn = mean_valid_loss_picnn.item()
                    makedirs(args.save)
                    bestParams_picnn = flow_picnn.state_dict()
                    torch.save({
                        'args': args,
                        'state_dict_ficnn': bestParams_ficnn,
                        'state_dict_picnn': bestParams_picnn,
                    }, os.path.join(args.save, strTitle + '_checkpt.pth'))
                else:
                    n_vals_wo_improve_picnn += 1
                log_message_valid += '  picnn no improve: {:d}/{:d}'.format(n_vals_wo_improve_picnn,
                                                                            args.early_stopping)

                logger.info(columns_valid)
                logger.info(log_message_valid)
                logger.info(["iter"] + columns_train)

            # update learning rate
            if n_vals_wo_improve_ficnn > args.early_stopping:
                if ndecs_ficnn > 2:
                    logger.info("early stopping engaged")
                    train_hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % strTitle))
                    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % strTitle))
                    exit(0)
                else:
                    update_lr_ficnn(optimizer1, n_vals_wo_improve_ficnn)
                    n_vals_wo_improve_ficnn = 0

            if n_vals_wo_improve_picnn > args.early_stopping:
                if ndecs_picnn > 2:
                    logger.info("early stopping engaged")
                    train_hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % strTitle))
                    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % strTitle))
                    exit(0)
                else:
                    update_lr_picnn(optimizer2, n_vals_wo_improve_picnn)
                    n_vals_wo_improve_picnn = 0

            itr += 1

    train_hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % strTitle))
    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % strTitle))
