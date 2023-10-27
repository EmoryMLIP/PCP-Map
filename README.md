# Partially Convex Potential Maps (PCP-Map)
Pytorch implementation of our Partially Convex Potential Maps. The model constructs partially 
monotone transport maps between probability measures as gradients of partially input convex 
neural networks (PICNN).

We test PCP-Map's performance on learning both the map between joint probability measures
(by constructing a block triangular transport map) with the addition of a fully input convex 
neural networks (FICNN) and the between conditional measures. Note that for some of the following 
scripts, users have to set the appropriate arguments in the parser based on to which dataset 
they are learning. Moreover, users in some occasions need to fill in the absolute paths of files.

Paper:
https://arxiv.org/abs/2310.16975

## UCI Tabular Datasets Experiments
Perform pilot runs to search for best hyperparameter combinations:

```
python pretrain_cond.py --data 'concrete' --input_x_dim 1 --input_y_dim 8
python pretrain_cond.py --data 'energy' --input_x_dim 1 --input_y_dim 9
python pretrain_cond.py --data 'yacht' --input_x_dim 1 --input_y_dim 6

python pretrain_joint.py --data 'parkinson' --input_x_dim 8 --input_y_dim 7
python pretrain_joint.py --data 'rd_wine' --input_x_dim 6 --input_y_dim 5
python pretrain_joint.py --data 'wt_wine' --input_x_dim 6 --input_y_dim 5
```

Perform experiments with the 10 best hyperparameter combinations from pilot runs:
```
python experiment_tab_cond.py
python experiment_tab_joint.py
```


Evaluate the trained model
```
python evaluate_cond.py
python evaluate_joint.py
```

## Stochastic Lotka-Volterra Experiment

Perform pilot runs to search for best hyperparameter combination:
```
python pretrain_cond.py --data 'lv' --input_x_dim 4 --input_y_dim 9
```

Perform training with the best hyperparameter combination:
```
python experiment_lv.py
```

Evaluate the trained model
```
python evaluate_lv.py
```

## 1D Shallow Water Equations Experiment
Perform pilot runs to search for best hyperparameter combination:
```
python pretrain_cond.py --data 'sw' --input_x_dim 100 --input_y_dim 3500 --theta_pca 1
```

Perform training with the best hyperparameter combination:
```
python experiment_sw.py
```

Evaluate the trained model
```
python evaluate_sw.py
```
