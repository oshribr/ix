"""Exports changepoints in CSV format for import into MongoDB.
"""

import sys
import os
import os.path
import argparse
import re
import pickle
import pandas
import numpy
import csv
import torch
from torch.autograd import Variable
from . import models
from . import __version__

INFIX = "changepoints"


def get_args(args=sys.argv[1:]):
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="monitor cp2csv {}: {}"
                    .format(__version__, __doc__))

    parser.add_argument("-v", "--version", action="version",
                        version=__version__)
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="output file, changepoint file with "
                        ".csv extension by default")
    parser.add_argument("-c", "--changepoints", type=str, default=None,
                        help="changepoints file in numpy format, "
                        "<stay-name>-<infix>.npy by default")
    parser.add_argument("-x", "--infix", type=str, default=INFIX,
                        help="infix of changepoint file, '{}' by default"
                        .format(INFIX))
    parser.add_argument("-y", "--output-infix", type=str, default=None,
                        help="infix of output file, input infix by default")
    parser.add_argument("-s", "--stay-id", type=str, default=None,
                        help="stay ID, by default taken "
                        "from stay file name")
    parser.add_argument("-t", "--no-types", action="store_true",
                        default=False,
                        help="disables type specifications in "
                             "column names, for mongoimport < 3.4")
    parser.add_argument("-m", "--model", type=str, default=None,
                        help="model for filtering irrelevant alerts")
    parser.add_argument("stay", help="dataframe with a single stay")
    args = parser.parse_args(args)

    # Fill in omitted arguments
    if args.stay_id is None:
        m = re.match(".*monitor-dataset-(.*)\.pkl", args.stay)
        if m is not None:
            args.stay_id = m.group(1)
        else:
            args.stay_id = args.stay
    if args.changepoints is None:
        name, ext = os.path.splitext(args.stay)
        args.changepoints = name + "-" + args.infix + ".npy"
    if args.output_infix is None:
        args.output_infix = args.infix
    if args.output is None:
        name, _ = os.path.splitext(args.stay)
        args.output = name + "-" + args.output_infix + ".csv"

    return args


def main():
    """
    """
    args = get_args()

    # Load the stay data frame
    with open(args.stay, "rb") as f:
        stay = pickle.load(f)

    # Load changepoints
    changepoints = numpy.load(args.changepoints)

    # define fields (column names)
    fields = ["time.date_ms(yyyy-MM-dd H:mm:ss)",
              "complication_name.string()",
              "prediction_probability.double()",
              "name.string()",
              "active.int32()",
              "stay_id.string()",
              "state.string()",
              "typ.string()",
              "model_id.int32()"]
    field_names = [re.match("(.*)\..*", name).group(1)
                   for name in fields]

    # define fixed/default values for fields
    COMPLICATION_NAME = "predict_deterioration_shock"
    ACTIVE = 1
    STATE = "No Trigger"
    TYP = "ThresholdAlert"
    MODEL_ID = 9999
    NAME = "ThresholdAlert-{}-{}".format(COMPLICATION_NAME, MODEL_ID)

    # Allocate placeholder for a row
    row = [None, COMPLICATION_NAME, None, NAME,
           ACTIVE, args.stay_id, STATE, TYP, MODEL_ID]
    idx = dict(zip(field_names, range(len(field_names))))

    # We only need the stay to compute the absolute time
    time_zero = stay.index[0]

    # if there is a model, let's load and initialize it
    model = None
    if args.model is not None:
        hidden_size = (changepoints.shape[1] - 2) // 2
        model = models.Alert(hidden_size)
        model.load_state_dict(torch.load(args.model))
        model.eval()

    with open(args.output, "w") as f:
        wtr = csv.writer(f)
        if args.no_types:
            wtr.writerow(field_names)
        else:
            wtr.writerow(fields)
        for i in range(len(changepoints)):
            if model is not None:
                if model(Variable(torch.from_numpy(changepoints[i, 2:]))) \
                   .data[0] <= 0.5:
                    # skip changepoint if the model rejects it
                    continue
            t = changepoints[i, 0]
            severity = changepoints[i, 1]
            row[idx["time"]] = time_zero + \
                pandas.Timedelta("{} minutes".format(t))
            row[idx["prediction_probability"]] = severity
            wtr.writerow(row)
