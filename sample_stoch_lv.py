import numpy as np
import scipy.io
from datasets.StochasticLotkaVolterra import StochasticLotkaVolterra

num_samples = 50000
stoch = StochasticLotkaVolterra()
x_test, y_test = stoch.sample_joint(num_samples)
data = np.concatenate((x_test, y_test), axis=1)

# TODO change file name if needed
file_name = 'data.mat'
scipy.io.savemat(file_name, {'data': data})
