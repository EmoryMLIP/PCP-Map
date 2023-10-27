# flow-based NPE using the sbi package https://www.mackelab.org/sbi/
import pickle
import numpy as np
import torch
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
from torch.distributions import MultivariateNormal
from shallow_water_model.prior import gaussian_kernel

# Hyperparams for training different from default.
num_hidden_features = 14
training_batch_size = 100

device = "cuda:0"

# Load data.
# TODO: change to correct path
dataset = np.load('.../PCP-Map/datasets/shallow_water_data3500.npz')['dataset']
# obtain theta projection matrix
parameters = dataset[:, :100]
cov = parameters.T @ parameters
L, V = torch.linalg.eigh(torch.tensor(cov))
Vx = V[:, -14:].numpy()
parameters_proj = parameters @ Vx
theta = torch.tensor(parameters_proj, dtype=torch.float32)
x = torch.tensor(dataset[:, 100:], dtype=torch.float32)

# Set up prior.
cov = gaussian_kernel(size=100, sigma=15, tau=100.0)
cov_tensor = torch.tensor(Vx.T @ cov @ Vx)
mean = torch.tensor(Vx.T @ (np.ones(100) * 10))
prior = MultivariateNormal(loc=mean.to(device), covariance_matrix=cov_tensor.to(device))

# Set up density estimator
print("Build posterior")
density_estimator_build_fun = posterior_nn(
    model="nsf",
    hidden_features=num_hidden_features,
    z_score_x=True,
)

# Run inference.
print("run inference")
inference = SNPE(
    prior=prior,
    density_estimator=density_estimator_build_fun,
    device=device,
    show_progress_bars=True,
)
with torch.autograd.set_detect_anomaly(True):
    density_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=training_batch_size
    )
posterior = inference.build_posterior(density_estimator)

# Save inference.
with open('.../PCP-Map/experiments/npe/sw_NPEpca.p', "wb") as fh:
    pickle.dump(dict(posterior=posterior, de=density_estimator), fh)
