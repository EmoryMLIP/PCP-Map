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

## Set-up

To run some files (e.g. the tests), you may need to add them to the path via
```
export PYTHONPATH="${PYTHONPATH}:."
```

## UCI Tabular Datasets Experiments

Due to copyright concerns, we did not provide the required datasets. 
They can be downloaded from the UC Irvine Machine Learning Repository. 

After downloading, please specify the paths to these 
datasets in file 'datasets/tabular_data.py' for loading purposes.

#### Perform pilot runs to search for best hyperparameter combinations:

```
python pretrain_cond.py --data 'concrete' --input_x_dim 1 --input_y_dim 8
python pretrain_cond.py --data 'energy' --input_x_dim 1 --input_y_dim 9
python pretrain_cond.py --data 'yacht' --input_x_dim 1 --input_y_dim 6

python pretrain_joint.py --data 'parkinson' --input_x_dim 8 --input_y_dim 7
python pretrain_joint.py --data 'rd_wine' --input_x_dim 6 --input_y_dim 5
python pretrain_joint.py --data 'wt_wine' --input_x_dim 6 --input_y_dim 5
```

#### Perform experiments with the 10 best hyperparameter combinations from pilot runs:
Before running following scripts, please change accordingly to correct file paths 
when loading datasets and hyperparameter combinations
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

#### Prepare training dataset:
Before running this script, change to correct absolute path for storing the dataset
```
python sample_stoch_lv.py
```

#### Perform pilot runs to search for best hyperparameter combination:
Before running following script, please change accordingly to correct file paths 
when loading datasets
```
python pretrain_cond.py --data 'lv' --input_x_dim 4 --input_y_dim 9
```

#### Perform training with the best hyperparameter combination:
Before running following script, please change accordingly to correct file paths 
when loading datasets and hyperparameter combinations
```
python experiment_lv.py
```

#### Evaluate the trained model
Before running following script, please change accordingly to correct file paths 
```
python evaluate_lv.py
```

## 1D Shallow Water Equations Experiment

#### Prepare training dataset:

Please change the "path_to_fcode" variable in "simulator.py" to the correct
absolute path to "shallow_water01_modified.f90".

Change the "--path_to_save" argument in "sample_shallow_water.py" to correct paths

```
for k in 1 2 3 4 5 6 7 8 9 10
do
  python sample_shallow_water.py --job_num $k
done
```

#### Process dataset (dimension reduction for observations 'y'):
Before running following script, please change accordingly to correct file paths 
when loading datasets
```
python shallow_water.py
```

The datasets used for the associated paper can be found through
https://drive.google.com/drive/folders/1ObuuATIEsC3z9d0S_WRp2lGZVOClu0Ip?usp=drive_link. Note
that the datasets here contain already projected, 3500-dimensional, observations "y".

#### Perform pilot runs to search for best hyperparameter combination:
```
python pretrain_cond.py --data 'sw' --input_x_dim 100 --input_y_dim 3500 --theta_pca 1
```

#### Perform training with the best hyperparameter combination:
Before running following script, please change accordingly to correct file paths 
when loading hyperparameter combinations
```
python experiment_sw.py
```

#### Evaluate the trained model
Before running following script, please change accordingly to correct file paths 
```
python evaluate_sw.py
```
