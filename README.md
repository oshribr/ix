# Fiddling with monitor data time series

AR-204 on JIRA. Learning from time series to predict future,
in short and long range. 

## Notebooks

Notebooks are in the `notebooks/` folder.

* sandbox.ipynb --- ignore this. 
* monitor-data.ipynb --- extracting raw monitor data from the dataset.
* visualize-predicts.ipynb --- visualization of predictions for a single stay.
* predict-stats.ipynb --- prediction performance evaluation.

## Comand-line utilities

Run `<command> -h` to get help on command-line options.

* extract --- extracts contiguous fragments from data for training
* prepare --- prepares the dataset as a tensor for training
* train --- trains a model on the dataset
* predict --- extends a stay with predictions and log-likelihoods of observations.

## Running experiments

### Overview

The stays are in `data/`. The order of preparing a data set for training is

* extract
* prepare

### Training

Use `train` to train a model. The depth must be greater than 1 for the model
to learn to predict more than a single step in the future and deal with
missing values.

### Prediction

Use `predict` to extend a frame with prediction. Models are in `models/`, data
folder should be a subdirectiory of `data/` (with the list of concept names as
the name) and the stay is a pickle file in `data/`. **The model and the data
must match.** Example:

    predict model/ALL-15.model data/ALL data/monitor-dataset-Ichilov_MICU_20194.pkl

### Helper scripts

* scripts/extract.sh --- extracts contiguous fragments (120 min by default). 
* scripts/train.sh --- trains a model.

For scripts, modify defaults by specifying environment variables on the command line.
For example:

    RATE=0.0001 scripts/train.sh

to set the training rate to 0.0001. Consult the scripts for available variables.
