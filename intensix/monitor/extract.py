"""Extracts contiguous time series fragments from stay data.
"""

import argparse
import sys
import os
import os.path
import pickle
import intensix.monitor.contiguous as cg
from . import __version__

MINROWS = 1


def get_args(args=sys.argv[1:]):
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="monitor extract {}: {}"
                    .format(__version__, __doc__))

    parser.add_argument("-v", "--version", action="version",
                        version=__version__)
    parser.add_argument("-n", "--minrows", type=int, default=MINROWS,
                        help="mininum number of contiguous rows, "
                             "{} by default".format(MINROWS))
    parser.add_argument("-c", "--columns", type=str,
                        help="columns to consider and extract")
    parser.add_argument("-o", "--output", type=str, default=".",
                        help="output folder")
    parser.add_argument("data", nargs="+",
                        help="dataframe with either a single stay "
                             "or with a dictionary of stays")

    args = parser.parse_args(args)
    return args


def oname(oloc, name, i):
    """Constructs output file name.  Arguments:
      oloc --- output folder,
      name --- input file name,
      i --- span index.
    Returns output file name.
    """
    return os.path.join(oloc,
                        "x-{}-{}".format(i, os.path.basename(name)))


def print_extracted(stay, df, i):
    """Reports extracted data.
    """
    print("{}[{:02d}]: {: 6d}".format(stay, i, len(df)))
    sys.stdout.flush()


def print_skipped(skipped):
    """Reports skipped stays.
    """
    if skipped % 100 == 0:
        print("{} skipped: no data".format(skipped))
        sys.stdout.flush()


def main():
    """
    """
    args = get_args()

    # Create output directory if does not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Extract column list, None means all columns
    columns = None
    if args.columns is not None:
        columns = [c.strip() for c in args.columns.split(",")]

    # Traverse data and extract contiguous time series
    skipped = 0
    for name in args.data:
        with open(name, "rb") as f:
            df = pickle.load(f)
        stay = os.path.splitext(os.path.basename(name))[0]
        i = None
        for i, xdf in enumerate(cg.extract(df,
                                           minrows=args.minrows,
                                           columns=columns)):
            with open(oname(args.output, name, i), "wb") as f:
                pickle.dump(xdf, f)
            print_extracted(stay, xdf, i)
        if i is None:
            skipped += 1
            print_skipped(skipped)
