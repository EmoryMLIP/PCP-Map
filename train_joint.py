import argparse
import os
import time
import datetime
import pandas as pd
import torch
from torch import distributions
from torch.utils.data import DataLoader
from lib.dataloader import dataloader
from datasets import tabular_data
from src.icnn import FICNN, PICNN
from src.triflow_ficnn import TriFlowFICNN
from src.triflow_picnn import TriFlowPICNN
from src.mmd import mmd
from lib.utils import count_parameters, makedirs, get_logger, AverageMeter

"""
argument parser for hyper parameters and model handling
"""

parser = argparse.ArgumentParser('TC-Flow')
parser.add_argument(
    '--data', choices=['wt_wine', 'rd_wine', 'parkinson'], type=str, default='rd_wine'
)
parser.add_argument('--input_x_dim',    type=int, default=6, help="input data convex dimension")
parser.add_argument('--input_y_dim',    type=int, default=5, help="input data non-convex dimension")
parser.add_argument('--feature_dim',    type=int, default=64, help="intermediate layer feature dimension")
parser.add_argument('--out_dim',        type=int, default=1, help="output dimension")
parser.add_argument('--num_layers_fi',  type=int, default=2, help="depth of FICNN network")
parser.add_argument('--num_layers_pi',  type=int, default=2, help="depth of PICNN network")

parser.add_argument('--clip',           type=bool, default=True, help="whether clipping the weights or not")
parser.add_argument('--tol',            type=float, default=1e-12, help="LBFGS tolerance")

parser.add_argument('--batch_size',     type=int, default=32, help="number of samples per batch")
parser.add_argument('--num_epochs',     type=int, default=1000, help="number of training steps")
parser.add_argument('--print_freq',     type=int, default=1, help="how often to print results to log")
parser.add_argument('--valid_freq',     type=int, default=20, help="how often to run model on validation set")
parser.add_argument('--early_stopping', type=int, default=10, help="early stopping of training based on validation")
parser.add_argument('--lr',             type=float, default=0.005, help="optimizer learning rate")
parser.add_argument("--lr_drop",        type=float, default=2.0, help="how much to decrease lr (divide by)")

parser.add_argument('--test_ratio',     type=float, default=0.10, help="test set ratio")
parser.add_argument('--valid_ratio',    type=float, default=0.10, help="validation set ratio")
parser.add_argument('--random_state',   type=int, default=42, help="random state for splitting dataset")

parser.add_argument('--save_test',      type=bool, default=True, help="save test numerics after training")
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


def load_data(dataset):
    if dataset == 'wt_wine':
        data = tabular_data.get_wt_wine()
    elif dataset == 'rd_wine':
        data = tabular_data.get_rd_wine()
    elif dataset == 'parkinson':
        data = tabular_data.get_parkinson()
    else:
        raise Exception("Dataset is Incorrect")
    return data


def evaluate_model(ficnn_model, picnn_model, data, batch_size, test_ratio, valid_ratio, random_state, input_y_dim,
                   input_x_dim, tol, bestParams_ficnn, bestParams_picnn):
    # load data
    dataset = load_data(data)
    dataset = tabular_data.process_data(dataset)
    dataset = tabular_data.normalize_data(dataset)
    dat = torch.tensor(dataset, dtype=torch.float32)
    _, _, testData, _ = dataloader(data, batch_size, test_ratio, valid_ratio, random_state)
    # load best model
    ficnn_model.load_state_dict(bestParams_ficnn)
    picnn_model.load_state_dict(bestParams_picnn)
    # load test data
    test_loader = DataLoader(
        testData,
        batch_size=batch_size, shuffle=True
    )
    # Obtain Test Metrics Numbers
    testLossMeter = AverageMeter()
    for _, test_sample in enumerate(test_loader):
        x_test = test_sample[:, input_y_dim:].requires_grad_(True).to(device)
        y_test = test_sample[:, :input_y_dim].requires_grad_(True).to(device)
        log_prob1 = ficnn_model.loglik_ficnn(y_test)
        log_prob2 = picnn_model.loglik_picnn(x_test, y_test)
        pb_mean_NLL = -(log_prob1 + log_prob2).mean()
        testLossMeter.update(pb_mean_NLL.item(), test_sample.shape[0])
    # Test Generated Samples
    sample_size = dat.shape[0]
    zy = torch.randn(sample_size, input_y_dim).to(device)
    zx = torch.randn(sample_size, input_x_dim).to(device)
    y_generated, _ = ficnn_model.g1(zy, tol=tol).detach().to(device)
    x_generated, _ = picnn_model.g2(zx, y_generated, tol=tol).detach().to(device)
    sample = torch.cat((y_generated, x_generated), dim=1)
    # calculate MMD statistic
    mean_max_dis = mmd(sample, dat)

    return testLossMeter.avg, mean_max_dis


"""
Training
"""

if __name__ == '__main__':

    # load data
    train_loader, valid_loader, _, train_size = dataloader(args.data, args.batch_size, args.test_ratio,
                                                           args.valid_ratio, args.random_state)

    # Establishing TC-Flows
    if args.clip is True:
        reparam = False
    else:
        reparam = True

    # Multivariate Gaussian as Reference
    prior_ficnn = distributions.MultivariateNormal(torch.zeros(args.input_y_dim).to(device),
                                                   torch.eye(args.input_y_dim).to(device))
    prior_picnn = distributions.MultivariateNormal(torch.zeros(args.input_x_dim).to(device),
                                                   torch.eye(args.input_x_dim).to(device))

    # establish TC-Flow
    ficnn = FICNN(args.input_y_dim, args.feature_dim, args.out_dim, args.num_layers_fi, reparam=reparam).to(device)
    picnn = PICNN(args.input_x_dim, args.input_y_dim, args.feature_dim, args.out_dim, args.num_layers_pi, reparam=reparam).to(device)

    flow_ficnn = TriFlowFICNN(prior_ficnn, ficnn).to(device)
    flow_picnn = TriFlowPICNN(prior_picnn, picnn).to(device)

    optimizer1 = torch.optim.Adam(flow_ficnn.parameters(), lr=args.lr)
    optimizer2 = torch.optim.Adam(flow_picnn.parameters(), lr=args.lr)

    strTitle = args.data + '_' + sStartTime + '_' + str(args.batch_size) + '_' + str(args.lr) + \
            '_' + str(args.num_layers_fi) + '_' + str(args.feature_dim)

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

    # training starts
    itr = 1
    total_itr = (int(train_size / args.batch_size) + 1) * args.num_epochs
    best_loss_ficnn = float('inf')
    best_loss_picnn = float('inf')
    bestParams_ficnn = None
    bestParams_picnn = None

    start_time = time.time()
    end = start_time
    makedirs(args.save)

    timeMeter = AverageMeter()

    for epoch in range(args.num_epochs):
        for i, sample in enumerate(train_loader):
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

            timeMeter.update(time.time() - end)

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

                valLossMeterFICNN = AverageMeter()
                valLossMeterPICNN = AverageMeter()

                for k, valid_sample in enumerate(valid_loader):
                    x_valid = valid_sample[:, args.input_y_dim:].requires_grad_(True).to(device)
                    y_valid = valid_sample[:, :args.input_y_dim].requires_grad_(True).to(device)
                    mean_valid_loss_ficnn = -flow_ficnn.loglik_ficnn(y_valid).mean()
                    mean_valid_loss_picnn = -flow_picnn.loglik_picnn(x_valid, y_valid).mean()
                    valLossMeterFICNN.update(mean_valid_loss_ficnn.item(), valid_sample.shape[0])
                    valLossMeterPICNN.update(mean_valid_loss_picnn.item(), valid_sample.shape[0])

                valid_hist.loc[len(valid_hist.index)] = [valLossMeterFICNN.avg, valLossMeterPICNN.avg]
                log_message_valid = '   {:9.3e}       {:9.3e} '.format(
                    valLossMeterFICNN.avg, valLossMeterPICNN.avg
                )

                if valLossMeterFICNN.avg < best_loss_ficnn:
                    n_vals_wo_improve_ficnn = 0
                    best_loss_ficnn = valLossMeterFICNN.avg
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

                if valLossMeterPICNN.avg < best_loss_picnn:
                    n_vals_wo_improve_picnn = 0
                    best_loss_picnn = valLossMeterPICNN.avg
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
                    logger.info("Training Time: {:} seconds".format(timeMeter.sum))
                    train_hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % strTitle))
                    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % strTitle))
                    if args.save_test is False:
                        exit(0)
                    NLL, MMD = evaluate_model(flow_ficnn, flow_picnn, args.data, args.batch_size, args.test_ratio,
                                              args.valid_ratio, args.random_state, args.input_y_dim, args.input_x_dim,
                                              args.tol, bestParams_ficnn, bestParams_picnn)
                    columns_test = ["batch_size", "lr", "width", "depth", "NLL", "MMD", "time", "iter"]
                    test_hist = pd.DataFrame(columns=columns_test)
                    test_hist.loc[len(test_hist.index)] = [args.batch_size, args.lr, args.feature_dim,
                                                           args.num_layers_pi, NLL, MMD, timeMeter.sum, itr]
                    testfile_name = '/TC-Flow/experiments/tabjoint/' + args.data + '_test_hist.csv'
                    if os.path.isfile(testfile_name):
                        test_hist.to_csv(testfile_name, mode='a', index=False, header=False)
                    else:
                        test_hist.to_csv(testfile_name, index=False)
                    exit(0)
                else:
                    update_lr_ficnn(optimizer1, n_vals_wo_improve_ficnn)
                    n_vals_wo_improve_ficnn = 0

            if n_vals_wo_improve_picnn > args.early_stopping:
                if ndecs_picnn > 2:
                    logger.info("early stopping engaged")
                    logger.info("Training Time: {:} seconds".format(timeMeter.sum))
                    train_hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % strTitle))
                    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % strTitle))
                    if args.save_test is False:
                        exit(0)
                    NLL, MMD = evaluate_model(flow_ficnn, flow_picnn, args.data, args.batch_size, args.test_ratio,
                                              args.valid_ratio, args.random_state, args.input_y_dim, args.input_x_dim,
                                              args.tol, bestParams_ficnn, bestParams_picnn)
                    columns_test = ["batch_size", "lr", "width", "depth", "NLL", "MMD", "time", "iter"]
                    test_hist = pd.DataFrame(columns=columns_test)
                    test_hist.loc[len(test_hist.index)] = [args.batch_size, args.lr, args.feature_dim,
                                                           args.num_layers_pi, NLL, MMD, timeMeter.sum, itr]
                    testfile_name = '/TC-Flow/experiments/tabjoint/' + args.data + '_test_hist.csv'
                    if os.path.isfile(testfile_name):
                        test_hist.to_csv(testfile_name, mode='a', index=False, header=False)
                    else:
                        test_hist.to_csv(testfile_name, index=False)
                    exit(0)
                else:
                    update_lr_picnn(optimizer2, n_vals_wo_improve_picnn)
                    n_vals_wo_improve_picnn = 0

            itr += 1
            end = time.time()

    print('Training time: %.2f secs' % timeMeter.sum)
    train_hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % strTitle))
    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % strTitle))
    if args.save_test is False:
        exit(0)
    NLL, MMD = evaluate_model(flow_ficnn, flow_picnn, args.data, args.batch_size, args.test_ratio,
                              args.valid_ratio, args.random_state, args.input_y_dim, args.input_x_dim,
                              args.tol, bestParams_ficnn, bestParams_picnn)
    columns_test = ["batch_size", "lr", "width", "depth", "NLL", "MMD", "time", "iter"]
    test_hist = pd.DataFrame(columns=columns_test)
    test_hist.loc[len(test_hist.index)] = [args.batch_size, args.lr, args.feature_dim, args.num_layers_pi, NLL, MMD,
                                           timeMeter.sum, itr]
    testfile_name = '/TC-Flow/experiments/tabjoint/' + args.data + '_test_hist.csv'
    if os.path.isfile(testfile_name):
        test_hist.to_csv(testfile_name, mode='a', index=False, header=False)
    else:
        test_hist.to_csv(testfile_name, index=False)
