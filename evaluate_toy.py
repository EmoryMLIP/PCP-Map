import argparse
import os
import torch
from torch import distributions
from datasets import toy_data
from src.plotter import plot4_toy
from src.icnn import FICNN, PICNN
from src.triflow_ficnn import TriFlowFICNN
from src.triflow_picnn import TriFlowPICNN

parser = argparse.ArgumentParser('PCPM')
parser.add_argument('--resume',         type=str, default="experiments/toy/moon_2022_11_13_13_22_33_checkpt.pth")

args = parser.parse_args()

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load best model
checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)

"""
Load Dataset
"""

dat = checkpt['args'].data
sample_size = checkpt['args'].sample_size
if dat == "moon":
    test_set = toy_data.get_moon(sample_size)
elif dat == "spiral":
    test_set = toy_data.get_spiral(sample_size)
elif dat == "circles":
    test_set = toy_data.get_circles(sample_size)
elif dat == "swiss":
    test_set = toy_data.get_swiss_roll(sample_size)
elif dat == "pinwheel":
    test_set = toy_data.get_pinwheel(sample_size)
elif dat == "8gauss":
    test_set = toy_data.get_8gauss(sample_size)
elif dat == "2spirals":
    test_set = toy_data.get_2spirals(sample_size)
elif dat == "checkerboard":
    test_set = toy_data.get_checkerboard(sample_size)
else:
    raise Exception("Dataset is Incorrect")


"""
Load Best Models
"""

print(checkpt['args'])
input_x_dim = checkpt['args'].input_x_dim
input_y_dim = checkpt['args'].input_y_dim
feature_dim = checkpt['args'].feature_dim
feature_y_dim = checkpt['args'].feature_y_dim
out_dim = checkpt['args'].out_dim
num_layers_fi = checkpt['args'].num_layers_fi
num_layers_pi = checkpt['args'].num_layers_pi
clip = checkpt['args'].clip
if clip is True:
    reparam = False
else:
    reparam = True

prior = distributions.MultivariateNormal(torch.zeros(1).to(device), torch.eye(1).to(device))
ficnn = FICNN(input_y_dim, feature_dim, out_dim, num_layers_fi, reparam=reparam).to(device)
picnn = PICNN(input_x_dim, input_y_dim, feature_dim, feature_y_dim, out_dim, num_layers_pi, reparam=reparam).to(device)
flow_ficnn = TriFlowFICNN(prior, ficnn)
flow_picnn = TriFlowPICNN(prior, picnn)

flow_ficnn.load_state_dict(checkpt["state_dict_ficnn"])
flow_picnn.load_state_dict(checkpt["state_dict_picnn"])
flow_ficnn = flow_ficnn.to(device)
flow_picnn = flow_picnn.to(device)

"""
plot figures and save
"""

sPath = os.path.join(checkpt['args'].save, 'figs', checkpt['args'].data + '_{:04d}.png')
z = torch.randn(sample_size, 2)
plot4_toy(flow_ficnn, flow_picnn, test_set, z, device, checkpt['args'].swap, checkpt['args'].tol, sPath,
          sTitle=dat + '_toy_visualizations', hidevals=True)
