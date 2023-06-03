# Partially Convex Potential Maps (PCPM)
Pytorch implementation of our Partially Convex Potential Maps. The model constructs monotone 
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

Perform pilot runs:

```
python pretrain_cond.py
python pretrain_joint.py
```

Perform experiments with best models:
```
python experiment_tab_cond.py
python experiment_tab_joint.py
```

Train a single PCPM model to learn the block triangular map:
```
python train_joint.py
```

Train a single PCPM model to learn the conditional map:
```
python train_cond.py
```

Evaluate the trained model
```
python evaluate_cond.py
python evaluate_joint.py
```

## Stochastic Lotka Volterra

Perform pilot run:
```
python pretrain_cond.py
```
with dataset as "lv".

Perform training with the best model:
```
python experiment_lv.py
```

Evaluate the trained model
```
python evaluate_lv.py
```
