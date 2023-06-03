import os
import numpy as np
import pandas as pd

loss_con = pd.read_csv('.../PCPM/experiments/tabcond/concrete_valid_hist.csv').to_numpy()
loss_enr = pd.read_csv('.../PCPM/experiments/tabcond/energy_valid_hist.csv').to_numpy()
loss_yat = pd.read_csv('.../PCPM/experiments/tabcond/yacht_valid_hist.csv').to_numpy()
param_con = pd.read_csv('.../PCPM/experiments/tabcond/concrete_params_hist.csv').to_numpy()
param_enr = pd.read_csv('.../PCPM/experiments/tabcond/energy_params_hist.csv').to_numpy()
param_yat = pd.read_csv('.../PCPM/experiments/tabcond/yacht_params_hist.csv').to_numpy()
loss_param_con = np.concatenate((param_con[:, 1:], loss_con[:, 1:]), axis=1)
loss_param_enr = np.concatenate((param_enr[:, 1:], loss_enr[:, 1:]), axis=1)
loss_param_yat = np.concatenate((param_yat[:, 1:], loss_yat[:, 1:]), axis=1)
unique_param_con = loss_param_con[np.unique(loss_param_con[:, :-1], return_index=True, axis=0)[1]]
unique_param_enr = loss_param_enr[np.unique(loss_param_enr[:, :-1], return_index=True, axis=0)[1]]
unique_param_yat = loss_param_yat[np.unique(loss_param_yat[:, :-1], return_index=True, axis=0)[1]]
unique_param_con = unique_param_con[unique_param_con[:, -1].argsort()]
unique_param_enr = unique_param_enr[unique_param_enr[:, -1].argsort()]
unique_param_yat = unique_param_yat[unique_param_yat[:, -1].argsort()]
param_con_list = unique_param_con[:10, :]
param_enr_list = unique_param_enr[:10, :]
param_yat_list = unique_param_yat[:10, :]


for i in range(10):
    for j in range(5):
        batch_size_con = int(param_con_list[i, 0])
        batch_size_enr = int(param_enr_list[i, 0])
        batch_size_yat = int(param_yat_list[i, 0])
        lr_con = param_con_list[i, 1]
        lr_enr = param_enr_list[i, 1]
        lr_yat = param_yat_list[i, 1]
        width_con = int(param_con_list[i, 2])
        width_enr = int(param_enr_list[i, 2])
        width_yat = int(param_yat_list[i, 2])
        num_layers_con = int(param_con_list[i, 3])
        num_layers_enr = int(param_enr_list[i, 3])
        num_layers_yat = int(param_yat_list[i, 3])

        os.system(
            "python train_cond.py --data 'concrete' --valid_freq 20 --early_stopping 10  --input_x_dim 1 --input_y_dim 8\
             --num_layers_pi " + str(num_layers_con) + " --feature_dim " + str(width_con) + " --batch_size " + str(
             batch_size_con) + " --lr " + str(lr_con) + " --save 'experiments/tabcond/concrete'"
        )

        os.system(
            "python train_cond.py --data 'energy' --valid_freq 20 --early_stopping 10 --input_x_dim 1 --input_y_dim 9\
             --num_layers_pi " + str(num_layers_enr) + " --feature_dim " + str(width_enr) + " --batch_size " + str(
             batch_size_enr) + " --lr " + str(lr_enr) + " --save 'experiments/tabcond/energy'"
        )

        os.system(
            "python train_cond.py --data 'yacht' --valid_freq 20 --early_stopping 10 --input_x_dim 1 --input_y_dim 6\
             --num_layers_pi " + str(num_layers_yat) + " --feature_dim " + str(width_yat) + " --batch_size " + str(
             batch_size_yat) + " --lr " + str(lr_yat) + " --save 'experiments/tabcond/yacht'"
        )
