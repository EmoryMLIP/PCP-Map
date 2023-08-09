from os import listdir
from os.path import join
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_data_swe(num_eigs, save=True):

    """Obtain Data"""

    path_to_sims = '.../PCP-Map/datasets/shallow_water_data/'
    files = sorted(f for f in listdir(path_to_sims) if f.endswith(".npz"))

    # Load data sequentially from files
    depth_profiles, z_vals = [], []
    for file in files:
        dd = np.load(join(path_to_sims, file))
        depth_profiles.append(dd["depth_profile"].squeeze())
        z_vals.append(dd["z_vals"].squeeze())

    # Reshape and turn into torch tensors
    depth_profiles = np.concatenate(depth_profiles, 0)
    z_vals = np.concatenate(z_vals, 0)[:, :, 1:]
    z_vals = torch.FloatTensor(z_vals)

    theta = depth_profiles    # parameters
    x = z_vals.reshape(len(z_vals), -1).to(device)    # reshaped observations (100k, 200x100)

    """Perform PCA"""

    # obtain the estimate covariance matrix
    x_cov_est = 1/(len(z_vals)-1) * x.T @ x
    # perform eigen decomposition
    L, V = torch.linalg.eigh(x_cov_est)
    # calculate percentage of spectrum captured
    percentage = 100*sum(L[x_cov_est.shape[0]-num_eigs:])/sum(L)
    print("Percentage of Spectrum with " + str(num_eigs) + " eigenvalues is: " + str(percentage.item()) + '%')

    """Project Data"""

    # get projection matrix
    Vs = V[:, x_cov_est.shape[0]-num_eigs:].to(device)
    # perform projection
    x = x.unsqueeze(-1)
    x_s = Vs.T @ x
    x_proj = x_s.squeeze(-1).cpu().numpy()
    dataset = np.concatenate((theta, x_proj), axis=1)
    # save data
    if save is True:
        np.savetxt("shallow_water_data" + str(num_eigs) + ".csv", dataset, delimiter=",")
    return Vs


def load_swdata(batch_size):
    # Load Data
    dataset = pd.read_csv('.../PCP-Map/datasets/shallow_water_data3500.csv', sep=',').to_numpy()
    # Split, Normalize
    train, valid = train_test_split(
        dataset,
        test_size=0.001,
        random_state=42
    )
    train_mean = np.mean(train, axis=0, keepdims=True)
    train_std = np.std(train, axis=0, keepdims=True)
    train = (train - train_mean) / train_std
    valid = (valid - train_mean) / train_std
    train_data = torch.tensor(train, dtype=torch.float32)
    valid_data = torch.tensor(valid, dtype=torch.float32)
    # Loader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size, shuffle=True
    )
    return dataset, train_loader, valid_data, train.shape[0]


if __name__ == '__main__':
    # prepare training dataset
    _ = create_data_swe(num_eigs=3500)
