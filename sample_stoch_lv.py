import numpy as np
import scipy.io
from datasets.StochasticLotkaVolterra import StochasticLotkaVolterra

num_samples = 50000
stoch = StochasticLotkaVolterra()
x_test, y_test = stoch.sample_joint(num_samples)
data = np.concatenate((x_test, y_test), axis=1)

# TODO specify file path
file_name = '.../PCP-Map/datasets/lv_data.mat'
scipy.io.savemat(file_name, {'data': data})
