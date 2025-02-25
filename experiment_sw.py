import os
import numpy as np
import pandas as pd

# TODO change to correct paths
loss = pd.read_csv('.../PCP-Map/experiments/tabcond/sw_valid_hist.csv').to_numpy()
param = pd.read_csv('.../PCP-Map/experiments/tabcond/sw_params_hist.csv').to_numpy()
loss_param = np.concatenate((param[:, 1:], loss[:, 1:]), axis=1)
unique_param = loss_param[np.unique(loss_param[:, :-1], return_index=True, axis=0)[1]]
unique_param = unique_param[unique_param[:, -1].argsort()]
param_list = unique_param[0, :]

batch_size = int(param_list[0])
lr = param_list[1]
width = int(param_list[2])
width_y = int(param_list[3])
num_layers = int(param_list[4])

os.system(
    "python train_cond.py --data 'sw' --valid_freq 70 --early_stopping 20  --input_x_dim 100 --input_y_dim 3500\
     --num_layers_pi " + str(num_layers) + " --feature_dim " + str(width) + " --feature_y_dim " + str(width_y) +
    " --batch_size " + str(batch_size) + " --lr " + str(lr) + " --theta_pca 1 --tol 1e-9 --save_test 1 \
    --save 'experiments/cond/sw'"
)
