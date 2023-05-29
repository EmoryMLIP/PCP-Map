import argparse
import torch
from torch import distributions
from lib.dataloader import dataloader
from src.icnn import PICNN
from src.triflow_picnn import TriFlowPICNN
from src.mmd import mmd

parser = argparse.ArgumentParser('TC-Flow')
parser.add_argument('--resume',      type=str, default="experiments/condition/concrete_2023_03_05_11_50_28_checkpt.pth")

args = parser.parse_args()

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load best model
checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)

if __name__ == '__main__':

    dataset = checkpt['args'].data
    batch_size = checkpt['args'].batch_size
    test_ratio = checkpt['args'].test_ratio
    valid_ratio = checkpt['args'].valid_ratio
    random_state = checkpt['args'].random_state
    _, _, test_data, _ = dataloader(dataset, batch_size, test_ratio, valid_ratio, random_state)

    # Load Best Models
    print(checkpt['args'])
    input_x_dim = checkpt['args'].input_x_dim
    input_y_dim = checkpt['args'].input_y_dim
    feature_dim = checkpt['args'].feature_dim
    out_dim = checkpt['args'].out_dim
    num_layers_pi = checkpt['args'].num_layers_pi
    clip = checkpt['args'].clip
    if clip is True:
        reparam = False
    else:
        reparam = True

    prior_picnn = distributions.MultivariateNormal(torch.zeros(input_x_dim).to(device), torch.eye(input_x_dim).to(device))
    picnn = PICNN(input_x_dim, input_y_dim, feature_dim, out_dim, num_layers_pi, reparam=reparam).to(device)
    flow_picnn = TriFlowPICNN(prior_picnn, picnn)

    flow_picnn.load_state_dict(checkpt["state_dict_picnn"])
    flow_picnn = flow_picnn.to(device)

    # Obtain test metrics numbers
    x_test = test_data[:, input_y_dim:].requires_grad_(True).to(device)
    y_test = test_data[:, :input_y_dim].requires_grad_(True).to(device)
    log_prob_picnn = flow_picnn.loglik_picnn(x_test, y_test)
    pb_mean_NLL = -log_prob_picnn.mean()
    print('Mean Conditional Negative Log Likelihood: {:.3e}'.format(pb_mean_NLL.item()))

    # Calculate MMD
    zx = torch.randn(test_data.shape[0], input_x_dim).to(device)
    x_generated, _ = flow_picnn.g2(zx, test_data[:, :input_y_dim].to(device), tol=checkpt['args'].tol)
    x_generated = x_generated.detach().to(device)
    mean_max_dis = mmd(x_generated, test_data[:, input_y_dim:])
    print('Maximum Mean Discrepancy: {:.3e}'.format(mean_max_dis))
