import argparse
import os
import torch
from torch import distributions
from torch.utils.data import DataLoader
from lib.dataloader import dataloader
from datasets import tabular_data
from src.plotter import plot4_tabular
from src.icnn import FICNN, PICNN
from src.triflow_ficnn import TriFlowFICNN
from src.triflow_picnn import TriFlowPICNN
from src.mmd import mmd
from lib.utils import AverageMeter

parser = argparse.ArgumentParser('PCPM')
parser.add_argument('--resume',         type=str, default="experiments/tabular/rd_wine_2022_11_09_20_17_42_checkpt.pth")

args = parser.parse_args()

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load best model
checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)


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


if __name__ == '__main__':

    dataset = checkpt['args'].data
    batch_size = checkpt['args'].batch_size
    test_ratio = checkpt['args'].test_ratio
    valid_ratio = checkpt['args'].valid_ratio
    random_state = checkpt['args'].random_state
    _, _, test_data, _ = dataloader(dataset, batch_size, test_ratio, valid_ratio, random_state)

    data = load_data(dataset)
    data = tabular_data.process_data(data)
    data = tabular_data.normalize_data(data)
    dat = torch.tensor(data, dtype=torch.float32)

    # Load Best Models
    print(checkpt['args'])
    input_x_dim = checkpt['args'].input_x_dim
    input_y_dim = checkpt['args'].input_y_dim
    feature_dim = checkpt['args'].feature_dim
    out_dim = checkpt['args'].out_dim
    num_layers_fi = checkpt['args'].num_layers_fi
    num_layers_pi = checkpt['args'].num_layers_pi
    clip = checkpt['args'].clip
    if clip is True:
        reparam = False
    else:
        reparam = True

    prior_ficnn = distributions.MultivariateNormal(torch.zeros(input_y_dim).to(device),
                                                   torch.eye(input_y_dim).to(device))
    prior_picnn = distributions.MultivariateNormal(torch.zeros(input_x_dim).to(device),
                                                   torch.eye(input_x_dim).to(device))
    ficnn = FICNN(input_y_dim, feature_dim, out_dim, num_layers_fi, reparam=reparam).to(device)
    picnn = PICNN(input_x_dim, input_y_dim, feature_dim, out_dim, num_layers_pi, reparam=reparam).to(device)
    flow_ficnn = TriFlowFICNN(prior_ficnn, ficnn)
    flow_picnn = TriFlowPICNN(prior_picnn, picnn)

    flow_ficnn.load_state_dict(checkpt["state_dict_ficnn"])
    flow_picnn.load_state_dict(checkpt["state_dict_picnn"])
    flow_ficnn = flow_ficnn.to(device)
    flow_picnn = flow_picnn.to(device)

    # load test data
    test_loader = DataLoader(
            test_data,
            batch_size=batch_size, shuffle=True
    )

    # Obtain Test Metrics Numbers
    testLossMeter = AverageMeter()

    for test_sample in test_loader:
        x_test = test_sample[:, input_y_dim:].requires_grad_(True).to(device)
        y_test = test_sample[:, :input_y_dim].requires_grad_(True).to(device)
        log_prob1 = flow_ficnn.loglik_ficnn(y_test)
        log_prob2 = flow_picnn.loglik_picnn(x_test, y_test)
        pb_mean_NLL = -(log_prob1 + log_prob2).mean()
        testLossMeter.update(pb_mean_NLL.item(), test_sample.shape[0])
    print('Mean Negative Log Likelihood: {:.3e}'.format(testLossMeter.avg))

    # Gaussian Pullback
    x_test_tot = test_data[:, input_y_dim:].requires_grad_(True).to(device)
    y_test_tot = test_data[:, :input_y_dim].requires_grad_(True).to(device)
    zy_approx = flow_ficnn.g1inv(y_test_tot).detach()
    zx_approx = flow_picnn.g2inv(x_test_tot, y_test_tot).detach()
    z = torch.cat((zy_approx, zx_approx), dim=1)

    # Test Generated Samples
    sample_size = dat.shape[0]
    zy = torch.randn(sample_size, input_y_dim).to(device)
    zx = torch.randn(sample_size, input_x_dim).to(device)
    y_generated, _ = flow_ficnn.g1(zy, tol=checkpt['args'].tol)
    y_generated = y_generated.detach().to(device)
    x_generated, _ = flow_picnn.g2(zx, y_generated, tol=checkpt['args'].tol)
    x_generated = x_generated.detach().to(device)
    sample = torch.cat((y_generated, x_generated), dim=1)

    # calculate MMD statistic
    mean_max_dis = mmd(sample, dat)
    print('Maximum Mean Discrepancy: {:.3e}'.format(mean_max_dis))

    # plot figures and save
    sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_{:03d}.png')
    plot4_tabular(dataset, z, sample, sPath, sTitle=dataset + '_visualizations', hidevals=True)
