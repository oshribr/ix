"""Trains a model for time series prediction on monitor data.
"""

import sys
import argparse
import os.path
import numpy
import numpy.random
import yaml
import torch
import torch.utils.data
import torch.optim
from torch.autograd import Variable
from . import models
from . import __version__
import pickle
# Default argument values
CONFIG = "train.yaml"
DEPTH = 2      # minimum depth for which the training works
ARCHITECTURE = "rnn"
TEST_FRACTION = 0.05
TRAIN_FRACTION = 1.0  # of remaining data
BATCH_SIZE = 128
NUM_EPOCHS = 100
BURNIN = 5
LEARNING_RATE = 0.001
PATIENCE = 5
EPSILON = 1E-8  # Adam's stabilizer


def get_args(args=sys.argv[1:]):
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="monitor train {}: {}"
                    .format(__version__, __doc__))

    parser.add_argument("-v", "--version", action="version",
                        version=__version__)
    parser.add_argument("-c", "--config", default=CONFIG,
                        help="configuration file, {} by default"
                        .format(CONFIG))
    parser.add_argument("-d", "--depth", type=int, default=DEPTH,
                        help="prediction depth, {} by default"
                        .format(DEPTH))
    parser.add_argument("-a", "--arch", default=ARCHITECTURE,
                        help="model architecture, {} by default"
                        .format(ARCHITECTURE))
    parser.add_argument("-b", "--batch-size", type=int,
                        default=BATCH_SIZE,
                        help="batch size, {} by default"
                        .format(BATCH_SIZE))
    parser.add_argument("-r", "--learning-rate", type=float,
                        default=LEARNING_RATE,
                        help="learning rate, {} by default"
                        .format(LEARNING_RATE))
    parser.add_argument("-e", "--epsilon", type=float,
                        default=EPSILON,
                        help="Adam's epsilon, {} by default"
                        .format(EPSILON))
    parser.add_argument("-t", "--test-fraction", type=float,
                        default=TEST_FRACTION,
                        help="fractions of data for validation "
                        "and testing, {} by default"
                        .format(TEST_FRACTION))
    parser.add_argument("-T", "--train-fraction", type=float,
                        default=TRAIN_FRACTION,
                        help="fraction of data for training, "
                        "{} of remaining data by default"
                        .format(TRAIN_FRACTION))
    parser.add_argument("-n", "--num-epochs", type=int,
                        default=NUM_EPOCHS,
                        help="number of training epochs, {} by default"
                        .format(NUM_EPOCHS))
    parser.add_argument("-u", "--burnin", type=int,
                        default=BURNIN,
                        help="minimum epochs before the model is saved, "
                        "{} by default"
                        .format(BURNIN))
    parser.add_argument("-p", "--patience", type=int,
                        default=PATIENCE,
                        help="number of epochs without progress "
                             "for early stopping, {} by default"
                        .format(PATIENCE))
    parser.add_argument('--no-cuda', default=False,
                        action="store_true",
                        help='disables CUDA')
    parser.add_argument("model", help="model")
    parser.add_argument("data", help="data")
    parser.add_argument("overrides", nargs="*",
                        help="config overrides, nested key are "
                        "concatenated by /")

    args = parser.parse_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda") if args.cuda else torch.device("cpu")
    print(f"args.device: {args.device}")
    return args


def read_config(fname, overrides):
    """Loads configuration and applies overrides.
    Components in compound keys are concatenated by "/".
    """
    with open(fname, "r") as f:
        config = yaml.full_load(f)

    # Apply config overrides
    for kv in overrides:
        k, v = kv.split("=")
        ks = k.split("/")
        v = yaml.load(v)
        c = config
        try:
            for k in ks[:-1]:
                c = c[k]
            # Coerce value to proper type
            v = type(c[ks[-1]])(v)
            c[ks[-1]] = v
        except KeyError:
            # misspelled key?
            print("ERROR: unrecognized override {}".format(kv),
                  file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            # wrong type or not a leaf
            print("ERROR: illegal override {}: {}".format(kv, e),
                  file=sys.stderr)
            sys.exit(1)

    return config


def load_model(args, cfg, input_size):
    """Loads stored model or creates a fresh one
    if does not exist or incompatible.
    """
    # Create storable model parameters from input and configuration
    params = {"arch": args.arch,
              "input_size": input_size,
              "kwargs": cfg[args.arch]}

    # Create the model object in which the model state, if available,
    # will be loaded
    model = models.make(args.arch, input_size, **params["kwargs"])

    if os.path.exists(args.model):
        loaded = torch.load(args.model)
        try:
            # Check that this is the same model type
            assert loaded["arch"] == args.arch, \
                "configured {}, but got {}" \
                .format(args.arch, loaded["arch"])

            # Input size must be the same
            assert params["input_size"] == loaded["input_size"], \
                "input size mismatch: data {}, model {}" \
                .format(params["input_size"], loaded["input_size"])

            state_dict = loaded["state_dict"]

            # Let's just try, if it fails we'll train from scratch
            model.load_state_dict(state_dict)
        except Exception as e:
            print("WARNING: Incompatible stored model {}: {}"
                  .format(args.model, e),
                  file=sys.stderr)

    return model, params


OPFNAME_TEMPLATE = "{},optim"


def load_optimizer(args, cfg, model):
    """Loads stored optimizer parameters.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 eps=args.epsilon)
    fname = OPFNAME_TEMPLATE.format(args.model)
    if os.path.exists(fname):
        state_dict = torch.load(fname)
        try:
            optimizer.load_state_dict(state_dict)
        except Exception as e:
            print("WARNING: incompatible stored "
                  "optimizer parameters {}: {}"
                  .format(fname, e),
                  file=sys.stderr)
    return optimizer


def save_model(args, model, params):
    """Saves model params into a file.
    """
    params["state_dict"] = model.state_dict()
    torch.save(params, args.model)


def save_optimizer(args, optimizer):
    """Saves optimizer parameters.
    """
    fname = OPFNAME_TEMPLATE.format(args.model)
    state_dict = optimizer.state_dict()
    torch.save(state_dict, fname)


def split_data(data, args):
    """Splits data in into train, validation, and tes sets.
    Returns the train set.
    """
    ntest = round(data.size(0) * args.test_fraction)
    nvald = ntest
    leftover = (data.size(0) - ntest - nvald) % args.batch_size
    # training may misbehave if leftover << batch_size;
    # move the leftover to the validation data
    nvald += leftover
    train_data = data[ntest:-nvald]
    validation_data = data[-nvald:]
    test_data = data[:ntest]
    return train_data, validation_data, test_data


def makexy(model, batch, depth):
    """Creates input and output from batch. Wraps
    model.makexy conveniently.
    """
    batch = batch.transpose(0, 1)
    x, y = model.makexy(batch, depth)
    x = Variable(x)
    y = Variable(y)
    return x, y


def main():
    """Loads data, trains model, saves model.
    """
    args = get_args()
    cfg = read_config(args.config, args.overrides)

    # Load data
    data = torch.from_numpy(numpy.load(args.data)).float()

    # Load the model
    model, model_parameters = load_model(args, cfg, data.size(2))
    model.to(args.device)

    # Split into train, validation, and test
    train_data, validation_data, test_data = split_data(data, args)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True)
    validation_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False)

    validation_loss = numpy.nan
    # Load stored or create a new optimizer
    optimizer = load_optimizer(args, cfg, model)

    # The model will be saved first when the loss is below the threshold
    train_loss = numpy.nan
    last_validation_loss = 1.  # roughly corresponds to unit variance

    # Train/test loop
    iepoch = 0
    last_iepoch_saved = numpy.nan  # last epoch the model was saved
    train_lossess, validation_lossess, test_lossess = [], [], []
    while True:
        # Validate and test the model
        with torch.no_grad():
            model.eval()
            loss = [0., 0.]
            for i, loader in enumerate([validation_loader, test_loader]):
                std = 0.
                for batch in loader:
                    batch = batch.to(args.device)
                    x, y = makexy(model, batch, args.depth)

                    # forward
                    result = model(x, args.depth)
                    preds = result[0]
                    std += preds.data[:, :, :, model.input_size:].abs() \
                        .mean(0) \
                        .mean(0) \
                        * len(batch)

                    # loss
                    batch_loss = model.loss(result, x, y)
                    loss[i] += batch_loss.data * len(batch)

                loss[i] /= len(loader.dataset)
                std /= len(loader.dataset)

        validation_loss, test_loss = loss

        print("epoch {:4d}: train loss: {:.6g} "
              "validation loss: {:.6g} test loss: {:.6g} "
              "average std: {:.6g} stds: {}"
              .format(iepoch, train_loss,
                      validation_loss, test_loss,
                      std.mean(), std))
        sys.stdout.flush()
        train_lossess.append(float(train_loss))
        validation_lossess.append(float(validation_loss))
        test_lossess.append(float(test_loss))

        losses, _ = os.path.splitext(args.model)
        losses = losses + '-losses.list'
        with open(losses, 'wb') as w:
            pickle.dump([train_lossess, validation_lossess, test_lossess], w)
        # Save the model
        if iepoch >= args.burnin and validation_loss <= last_validation_loss:
            if os.path.exists(args.model):
                os.rename(args.model, args.model + "~")
            if args.cuda:
                model.cpu()
            save_model(args, model, model_parameters)
            save_optimizer(args, optimizer)
            if args.cuda:
                model.cuda()
            last_validation_loss = validation_loss
            last_iepoch_saved = iepoch

        if iepoch == args.num_epochs:
            break
        if iepoch - last_iepoch_saved == args.patience:
            print("WARNING: no progress, stopping early",
                  file=sys.stderr)
            break
        iepoch += 1

        # Train the model
        model.train()
        train_loss = 0.
        ibatch = 0  # batches are selected based on args.train_fraction
        for batch in train_loader:
            optimizer.zero_grad()

            if numpy.random.random() > args.train_fraction:
                continue

            batch = batch.to(args.device)

            x, y = makexy(model, batch, args.depth)

            # forward
            result = model(x, args.depth)
            preds = result[0]

            # loss
            batch_loss = model.loss(result, x, y)
            train_loss += batch_loss.data * len(batch)

            # backward
            batch_loss.backward()
            optimizer.step()

            print("batch {:4d}: train loss: {:.6g}"
                  .format(ibatch, batch_loss.data))
            sys.stdout.flush()  # to see the progress when redirected

            ibatch += 1

        train_loss /= len(train_loader.dataset) * args.train_fraction
