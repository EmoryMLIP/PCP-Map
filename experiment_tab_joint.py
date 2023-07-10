import os
import numpy as np
import pandas as pd

loss_pk = pd.read_csv('.../PCP-Map/experiments/tabjoint/parkinson_valid_hist.csv').to_numpy()
loss_rd = pd.read_csv('.../PCP-Map/experiments/tabjoint/rd_wine_valid_hist.csv').to_numpy()
loss_wt = pd.read_csv('.../PCP-Map/experiments/tabjoint/wt_wine_valid_hist.csv').to_numpy()
param_pk = pd.read_csv('.../PCP-Map/experiments/tabjoint/parkinson_params_hist.csv').to_numpy()
param_rd = pd.read_csv('.../PCP-Map/experiments/tabjoint/rd_wine_params_hist.csv').to_numpy()
param_wt = pd.read_csv('.../PCP-Map/experiments/tabjoint/wt_wine_params_hist.csv').to_numpy()
loss_param_pk = np.concatenate((param_pk[:, 1:], loss_pk[:, 1:]), axis=1)
loss_param_rd = np.concatenate((param_rd[:, 1:], loss_rd[:, 1:]), axis=1)
loss_param_wt = np.concatenate((param_wt[:, 1:], loss_wt[:, 1:]), axis=1)
unique_param_pk = loss_param_pk[np.unique(loss_param_pk[:, :4], return_index=True, axis=0)[1]]
unique_param_rd = loss_param_rd[np.unique(loss_param_rd[:, :4], return_index=True, axis=0)[1]]
unique_param_wt = loss_param_wt[np.unique(loss_param_wt[:, :4], return_index=True, axis=0)[1]]
unique_param_pk = unique_param_pk[unique_param_pk[:, -1].argsort()]
unique_param_rd = unique_param_rd[unique_param_rd[:, -1].argsort()]
unique_param_wt = unique_param_wt[unique_param_wt[:, -1].argsort()]
param_pk_list = unique_param_pk[:10, :]
param_rd_list = unique_param_rd[:10, :]
param_wt_list = unique_param_wt[:10, :]


for i in range(10):
    for j in range(5):
        batch_size_pk = int(param_pk_list[i, 0])
        batch_size_rd = int(param_rd_list[i, 0])
        batch_size_wt = int(param_wt_list[i, 0])
        lr_pk = param_pk_list[i, 1]
        lr_rd = param_rd_list[i, 1]
        lr_wt = param_wt_list[i, 1]
        width_pk = int(param_pk_list[i, 2])
        width_rd = int(param_rd_list[i, 2])
        width_wt = int(param_wt_list[i, 2])
        width_y_pk = int(param_pk_list[i, 3])
        width_y_rd = int(param_rd_list[i, 3])
        width_y_wt = int(param_wt_list[i, 3])
        num_layers_pk = int(param_pk_list[i, 4])
        num_layers_rd = int(param_rd_list[i, 4])
        num_layers_wt = int(param_wt_list[i, 4])

        os.system(
            "python train_joint.py --data 'parkinson' --input_x_dim 8 --input_y_dim 7  --num_layers_fi " + str(
             num_layers_pk) + " --num_layers_pi " + str(num_layers_pk) + " --feature_dim " + str(width_pk) +
            " --feature_y_dim " + str(width_y_pk) + " --batch_size " + str(batch_size_pk) + " --lr " + str(lr_pk) +
            " --save 'experiments/tabjoint/parkinson'"
        )

        os.system(
            "python train_joint.py --data 'rd_wine' --num_layers_fi " + str(num_layers_rd) + " --num_layers_pi " + str(
             num_layers_rd) + " --feature_dim " + str(width_rd) + " --feature_y_dim " + str(width_y_rd) +
            " --batch_size " + str(batch_size_rd) + " --lr " + str(lr_rd) + " --save 'experiments/tabjoint/red'"
        )

        os.system(
            "python train_joint.py --data 'wt_wine' --num_layers_fi " + str(num_layers_wt) + " --num_layers_pi " + str(
             num_layers_wt) + " --feature_dim " + str(width_wt) + " --feature_y_dim " + str(width_y_wt) +
            " --batch_size " + str(batch_size_wt) + " --lr " + str(lr_wt) + " --save 'experiments/tabjoint/white'"
        )
