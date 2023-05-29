# Partially Convex Potential Maps (PCPM)
Pytorch implementation of our triangular convex flows model. The model constructs monotone 
block triangular transport maps between probability measures as gradients of ICNNs.

## Toy problems

Train a toy example
```
python train_toy.py
```

Plot results of a pre-trained model
```
python evaluate_toy.py
```

## Small tabular dataset experiment

Train TC-Flow over joint distribution
```
python train_highd.py
```

Evaluate the trained model
```
python evaluate_tabular.py
```

Train TC-Flow over conditional distribution
```
python train_cond.py
```

Evaluate the trained model
```
python evaluate_cond.py
```

Run TC-Flow random hyperparameter experiment
```
python experiment_tab_joint.py
python experiment_tab_cond.py
```

## Stochastic Lotka Volterra
Train TC-Flow on StochLV
```
python train_lv.py
```

Evaluate the trained model
```
python evaluate_lv.py
```

Run TC-Flow random hyperparameter experiment
```
python experiment_lv.py
```