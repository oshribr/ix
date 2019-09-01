"""Prepares data tensor for time series prediction training.
The data set is a three-dimensional tensor (samples,
time-series, features).

We intend to use it with Torch, but for more broad compatibility
will use a numpy tensor.
"""

import sys
import argparse
import os
import re
import pickle
import numpy
from . import __version__

LENGTH = 60  # default time series length
PATTERN = "x-.*\.pkl"


def get_args(args=sys.argv[1:]):
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="monitor prepare {}: {}"
                    .format(__version__, __doc__))

    parser.add_argument("-v", "--version", action="version",
                        version=__version__)
    parser.add_argument("-l", "--length", type=int, default=LENGTH,
                        help="time series length, {} by default"
                        .format(LENGTH))
    parser.add_argument("-p", "--pattern", type=str, default=PATTERN,
                        help="data file name pattern, '{}' by default"
                        .format(PATTERN))
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="output folder, data folder by default.")
    parser.add_argument("data", help="data folder")

    args = parser.parse_args(args)

    if args.output is None:
        args.output = args.data

    return args


def get_stay_names(data, pattern):
    """Retrieves the list of stay file names.
    Arguments:
      data --- data folder
      pattern --- data file name pattern
    Returns the list (without the folder name).
    """
    names = [name for name in os.listdir(data)
             if re.match(pattern, name)]
    return names


def get_column_names(data, name):
    """Extracts column names from the data in the first file.
    Arguments:
      data --- data folder
      name --- stay file name
    Returns the list of column names.
    """
    with open(os.path.join(data, name), "rb") as f:
        df = pickle.load(f)
    return list(df.columns)


def concatenate_stays(data, names, columns, length):
    """Concatenates data from all stays into a single
    ndarray, truncating each file's data into an integral
    number of series lengths. Arguments:
      data --- data folder
      names --- list of stay file names
      columns --- list of column names
      length --- sample length
    Returns the array.
    """
    stays = []
    for name in names:
        with open(os.path.join(data, name), "rb") as f:
            df = pickle.load(f)
        stay = df.as_matrix(columns)
        stay = stay[:stay.shape[0] // length * length, :]
        stays.append(stay)
    return numpy.concatenate(stays, axis=0)


def standardize(series):
    """Standardizes a series columnwise. Arguments:
      series --- an ndarray.
    Returns the standardized series, and mean, std
    for each column.
    """
    # We could use sklearn.preprocessing.Scaler for this,
    # but this is so simple that pulling scikit-learn would
    # be an overkill.
    m = series.mean(axis=0, keepdims=1)
    s = series.std(axis=0, keepdims=1)
    return (series - m) / s, numpy.concatenate([m, s], axis=0)


def reshape(stdstays, length):
    """Reshape the series to a dataset.
    The input dimensions are time point, feature.
    The output dimensions are sample, time point, feature.
    Arguments:
      args --- command-line arguments
      stdstays --- input array.
    Returns reshaped array.
    """
    return stdstays.reshape((-1, length, stdstays.shape[-1]))


def store(output, columns, scale, dataset):
    """Stores the dataset in the output folder.
    Arguments:
      output --- the output folder
      columns --- list of columns
      scale --- standard scale
      dataset --- the data array
    """
    with open(os.path.join(output, "columns"), "w") as f:
        for name in columns:
            print(name, file=f)
    with open(os.path.join(output, "scale"), "wb") as f:
        numpy.save(f, scale, allow_pickle=False)
    with open(os.path.join(output, "array"), "wb") as f:
        numpy.save(f, dataset, allow_pickle=False)


def main():
    """Command-line interface for data tensor preparation.
    Assumes every stay is a separate file. Stay collections
    are not currently supported.
    """
    args = get_args()

    # First, we collect the data file names, we will need
    # the list more than a single time
    names = get_stay_names(args.data, args.pattern)
    assert names, "the name list is empty: " \
                  "folder='{}', pattern='{}'" \
                  .format(args.data, args.pattern)

    # Then, we read the first file from the data set
    # to obtain the column names
    columns = get_column_names(args.data, names[0])

    # With the column names at hand, we concatenate all
    # of the data into a single array, truncating series
    # to an integral number of lengths.
    allstays = concatenate_stays(args.data, names, columns,
                                 args.length)

    # We standardize the data, and keep the mean and std
    # for each column to apply the model to future data
    stdstays, scale = standardize(allstays)

    # We now reshape the series to obtain the dataset
    dataset = reshape(stdstays, args.length)

    # Finally, time to serialize the data set. Three files
    # are created:
    #   columns --- the list of column names
    #   scale --- the matrix of mean and std for each column
    #   array --- the data set as three dimensional numpy array
    # For more compatibility, we disallow pickle and store column
    # names as plain text.
    store(args.output, columns, scale, dataset)
