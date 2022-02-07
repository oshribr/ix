"""Accepts a trained model (along with the list of columns and
the scaling) and a stay, as a pandas frame, and produces a frame
augmented with predictions for time step with the specified delay.

The structure of the frame is that for every time step
and feature X there are columns X_mean, X_std, X_nll.
"""

import sys
import os
import os.path
import argparse
import yaml
import pickle
import pandas
import numpy
import torch
from torch.autograd import Variable
from . import models
from . import __version__

DELAY = 1
INFIX = "predict"
MISSING = "distribution"

def get_args(args=sys.argv[1:]):
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="monitor predict {}: {}"
                    .format(__version__, __doc__))

    parser.add_argument("-v", "--version", action="version",
                        version=__version__)
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="output file, "
                        "<input name>-<infix>.<input extension> "
                        "by default")
    parser.add_argument("-r", "--dictionary", type=str, default=None,
                        help="output file directory, "
                        "stay file directory by default")
    parser.add_argument("-q", "--quiet", action="store_true", default=False,
                        help="print less on the console")
    parser.add_argument("-x", "--infix", type=str, default=INFIX,
                        help="infix of output file, '{}' by default"
                        .format(INFIX))
    parser.add_argument("-d", "--delay", type=int, default=DELAY,
                        help="prediction delay, {} by default"
                        .format(DELAY))
    parser.add_argument("-m", "--missing", type=str, default=MISSING,
                        choices=["distribution", "mean", "sampling"],
                        help="fill missing points strategy, {} by default"
                        .format(MISSING))
    parser.add_argument("model", help="model file")
    parser.add_argument("data",
                        help="data folder, must contain "
                        "columns, scale")
    parser.add_argument("stay", help="dataframe with a single stay")
    args = parser.parse_args(args)

    return args


def compute_predictions(model, obs, depth, missing):
    """Computes predictions based on the observations.
    Returns the predictions and their nlls.
    """
    # Convert observations into a batch
    obs = obs.reshape((obs.shape[0], 1, obs.shape[1]))
    obs = torch.from_numpy(obs).float()

    # Extract the input
    x, y = model.makexy(obs, depth)
    x = Variable(x)
    y = Variable(y)

    # Run the model forward through the input
    preds = model(x, depth, missing=missing)[0]

    # compute LML
    nlls = model.pred_nlls(preds, y)

    # Convert output back into a numpy array,
    # squeezing batch dimension and keeping only
    # the last prediction
    preds = preds.data.numpy()
    preds = preds[:, 0, -1, :]
    nlls = nlls.data.numpy()
    nlls = nlls[:, 0, -1, :]

    return preds, nlls


def sync_preds(preds, delay):
    """rearranges predictions so that predictions in the same
    row are for the same time step.
    """
    # Create padding with priors where predictions are unavailable
    padding = numpy.ndarray((delay, preds.shape[1]))

    # Fill the padding with priors
    padding[:, :preds.shape[1] // 2] = 0.
    padding[:, preds.shape[1] // 2:] = 1.

    # Prepend padding  to the predictions
    preds = numpy.concatenate([padding, preds], axis=0)

    return preds


def sync_nlls(nlls, delay):
    """rearranges nlls so that nlls in the same
    row are for the same time step.
    """
    # Create padding with nans where predictions are unavailable
    padding = numpy.ndarray((delay, nlls.shape[1]))

    # Fill the padding with priors
    padding[:, :] = numpy.nan

    # Prepend padding  to the nllsictions
    nlls = numpy.concatenate([padding, nlls], axis=0)

    return nlls


def main():
    """Loads model and stay, extends stay with prediction.
    """
    args = get_args()

    # Load the model
    params = torch.load(args.model)
    model = models.make(params["arch"],
                        params["input_size"],
                        **params["kwargs"])
    model.load_state_dict(params["state_dict"])
    del params["state_dict"]
    if not args.quiet:
        print(yaml.dump(params), file=sys.stderr)
    model.eval()

    # Load the data scale and column list
    scale = numpy.load(os.path.join(args.data, "scale"))
    with open(os.path.join(args.data, "columns"), "r") as f:
        columns = list(name.strip() for name in f)
    assert params["input_size"] == len(columns), \
        "ERROR: incompatible model: has {} columns, " \
        "but data has {}" \
        .format(params["input_size"], len(columns))

    # Load the stay data frame
    with open(args.stay, "rb") as f:
        stay = pickle.load(f)

    # Extract the time series, including missing data
    obs = stay[columns].to_numpy()

    # Normalize the time series
    nobs = (obs - scale[0])/scale[1]

    # Compute predictions for all data points
    # Reshape the matrix as a batch
    with torch.no_grad():
        npreds, nlls = compute_predictions(model, nobs, args.delay,
                                           args.missing)
    print("stay: {}\tavg std: {:0=.6f}\tavg NLL: {: .6g}"
          .format(args.stay,
                  numpy.abs(npreds[:, npreds.shape[1] // 2:]).mean(),
                  nlls[numpy.logical_not(numpy.isnan(nlls))]
                  .mean()))
    sys.stdout.flush()

    # Rearrange predictions so that predictions in the same row
    # are for the same time step but from different past points
    # of view, same for NLLs
    npreds = sync_preds(npreds, args.delay)
    nlls = sync_nlls(nlls, args.delay)

    # Scale predictions into original, unnormalized space
    preds = numpy.concatenate(
        (npreds[:, :npreds.shape[1] // 2]*scale[1] + scale[0],
         numpy.abs(npreds[:, npreds.shape[1] // 2:])*scale[1]),
        axis=1)

    # Merge predictions into the data frame
    for icol, colname in enumerate(columns):
        for j in range(preds.shape[1]):
            stay = stay.assign(
                **{"{}_mean".format(colname):
                   preds[:, icol],
                   "{}_std".format(colname):
                   preds[:, preds.shape[1] // 2 + icol],
                   "{}_nll".format(colname):
                   nlls[:, icol]})

    # Store the extended dataframe
    if args.output is None:
        name, ext = os.path.splitext(args.stay)
        args.output = name + "-" + args.infix + ext
    if args.dictionary is not None:
        _, name = os.path.split(args.output)
        args.output = os.path.join(args.dictionary, name)
    with open(args.output, "wb") as f:
        pickle.dump(stay, f)
