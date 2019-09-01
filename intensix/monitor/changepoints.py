"""Finds monitor changepoints. A changepoint is detected
according to divergence between short-term and long-term
prediction.  A changepoint record consists of time,
divergence, and hidden states at the beginning and
the end of the window.

The changepoints can be further filtered by a model which
accepts the hidden states as input and classifies
into either relevant or irrelevant changepoints. Relevant
changepoints should result in alerts.
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

INFIX = "changepoints"

WINDOW = 15
HORIZON = 1

CHANGEPOINT_THRESHOLD = 0.5


def get_args(args=sys.argv[1:]):
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="monitor changepoints {}: {}"
                    .format(__version__, __doc__))

    parser.add_argument("-v", "--version", action="version",
                        version=__version__)
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="output file, "
                        "<input name>-<infix>.npy by default")
    parser.add_argument("-q", "--quiet", action="store_true",
                        default=False,
                        help="print less on the console")
    parser.add_argument("-x", "--infix", type=str, default=INFIX,
                        help="infix of output file, '{}' by default"
                        .format(INFIX))
    parser.add_argument("-w", "--window", type=int, default=WINDOW,
                        help="changepoint window, {} by default"
                        .format(WINDOW))
    parser.add_argument("-z", "--horizon", type=int, default=HORIZON,
                        help="prediction horizon, {} by default"
                        .format(HORIZON))
    parser.add_argument("-t", "--changepoint-threshold", type=float,
                        default=CHANGEPOINT_THRESHOLD,
                        help="threshold for detecting changepoints, "
                        "{} by default"
                        .format(CHANGEPOINT_THRESHOLD))
    parser.add_argument("model", help="model file")
    parser.add_argument("data",
                        help="data folder, must contain "
                        "columns, scale")
    parser.add_argument("stay", help="dataframe with a single stay")
    args = parser.parse_args(args)

    return args


def as_batch(episode):
    """Reshapes episode as a batch.
    """
    episode = episode.reshape((episode.shape[0], 1, episode.shape[1]))
    batch = torch.from_numpy(episode).float()
    return batch


def track_episode(model, episode, depth, x0, h):
    """Runs the model through the episode, twice the window long.
    Returns
      * next episode predictions,
      * hidden state at the beginning of the next episode.
    """
    # Extract the input
    x = model.makex(episode, depth=0)  # use all of the episode to predict
    x = Variable(x, volatile=True)

    # Run the model forward through the input
    return model(x, depth, x0, h, missing=True)[:2]


def sigmoid(x):
    """sigmoid for bounding KLs
    """
    return -1 + 2/(1 + torch.exp(-x))


def main():
    """Loads model and stay, generates a numpy array of changepoints.
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
    observations = stay[columns].as_matrix()

    # Normalize the time series
    observations = (observations - scale[0])/scale[1]

    # Go through data windows and generate predictions with +window
    # delay, as well as keep track of hidden state at every time point

    # Prime the tracking
    t = 0
    x0 = None
    hprev = None
    prevs = None
    changepoints = []
    while True:
        # Get the next episode
        episode = as_batch(observations[t:t + args.window])
        t += len(episode)
        # Compute prediction for the next episode
        preds, h = track_episode(model, episode, args.horizon, x0, hprev)

        x0 = preds[-1, :, 0]  # first of last step, to chain episodes
        if t == len(observations):
            break
        if prevs is not None:
            kls = models.compute_kls(
                      torch.cat([prevs[-args.horizon:, :,
                                       args.horizon - 1],
                                 preds[:-args.horizon, :,
                                       args.horizon - 1]],
                                dim=0),
                      prevs[:, :, -1])
            severities = sigmoid(kls.data[:, 0].mean(dim=1))
            color = 'tab:green'
            lw = 1
            max_severity = severities.max()
            if max_severity >= args.changepoint_threshold:
                if not args.quiet:
                    print("time: {} severity: {:.4f}"
                          .format(stay.index[0] +
                                  pandas.Timedelta("{} minutes"
                                                   .format(t)),
                                  max_severity))
                changepoint = torch.cat([torch.Tensor([t, max_severity]),
                                         hprev[-1, 0].data,
                                         h[-1, 0].data],
                                        dim=0)
                changepoints.append(changepoint)

        if not args.quiet:
            print("{:06d}/{:06d}" .format(t, len(observations)),
                  end="\r")
        prevs = preds
        hprev = h

    # All augmented changepoints for the stay
    changepoints = torch.stack(changepoints, dim=0)
    # Store as numpy array for broader compatibility
    changepoints = changepoints.numpy()

    # Save the changepoints
    if args.output is None:
        name, ext = os.path.splitext(args.stay)
        args.output = name + "-" + args.infix + ".npy"
    numpy.save(args.output, changepoints)
