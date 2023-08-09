# Partially Convex Potential Maps (PCP-Map)
Pytorch implementation of our Partially Convex Potential Maps. The model constructs partially 
monotone transport maps between probability measures as gradients of partially input convex 
neural networks (PICNN).

## UCI Tabular Datasets Experiments
We test PCP-Map's performance on learning both the map between joint probability measures
(by constructing a block triangular transport map) with the assist of a fully input convex 
neural networks (FICNN) and the between conditional measures. Note that for some of the following 
scripts, users have to set the appropriate arguments in the parser according to which dataset 
they are learning.

Perform pilot runs to search for best hyperparameter combinations:

```
python pretrain_cond.py
python pretrain_joint.py
```

Perform experiments with the 10 best hyperparameter combinations from pilot runs:
```
python experiment_tab_cond.py
python experiment_tab_joint.py
```

Train a single PCPM model:
```
python train_joint.py
python train_cond.py
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
python pretrain_cond.py --data 'sw' --input_x_dim 100 --input_y_dim 3500
```

Perform training with the best hyperparameter combination:
```
python experiment_sw.py
```

Evaluate the trained model
```
python evaluate_sw.py
```
