"""Evaluates the ability of the prediction to predict a tag.
"""

import sys
import argparse
import re
import yaml
import pickle
import numpy
from . import __version__

# For evaluation, first and last SKIP minutes are ignored.

SKIP = 60

# If the stay contains the tag of interest, alerts between FROM_
# and UNTIL_ are positive, any other alerts are neutral (that is,
# ignored).
#
# If the stay does not contain the tag, any alert is negative.

FROM_ = 8*60
UNTIL_ = 2*60

# To tame the heavy tail, we smoothen the measurements through
# a rolling window.

WINDOW = 15

# NLL threshold should be learned from labeled data. The default
# is just a ballpark value.

THRESHOLD = 2.


def get_args(args=sys.argv[1:]):
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="monitor evaluate {}: {}"
                    .format(__version__, __doc__))

    parser.add_argument("-v", "--version", action="version",
                        version=__version__)
    parser.add_argument("-s", "--skip", type=int, default=SKIP,
                        help="time steps to skip in the beginning, "
                             "{} by default"
                             .format(SKIP))
    parser.add_argument("-f", "--from-", type=int, default=FROM_,
                        help="beginning of prediction interval, "
                             "{} by default"
                             .format(FROM_))
    parser.add_argument("-u", "--until-", type=int, default=UNTIL_,
                        help="end of prediction interval, "
                             "{} by default"
                             .format(UNTIL_))
    parser.add_argument("-w", "--window", type=int, default=WINDOW,
                        help="rolling window width, "
                             "{} by default"
                             .format(WINDOW))
    parser.add_argument("-t", "--threshold", type=int, default=THRESHOLD,
                        help="alert threshold, {} by default"
                             .format(THRESHOLD))
    parser.add_argument("-c", "--concepts", default=None,
                        help="concepts to evaluate, all by default")
    parser.add_argument("stay", help="stay augmented with predictions")
    parser.add_argument("tag", help="tag to predict")
    args = parser.parse_args(args)

    return args


def get_nllnames(stay, concepts):
    """Retrieves column names to evaluate on.
    """
    if concepts is None:
        nllnames = [n for n in stay.columns if re.match(".*_nll", n)]
    else:
        nllnames = ["{}_nll".format(c) for c in concepts.split(",")]
        assert nllnames[0] in stay, \
            "no prediction for concept {}".format(concept)

    return nllnames


def main():
    """Loads a stay augmented with predictions, computes prediction
    accuracy and outputs the evaluation.
    """
    args = get_args()

    # Load the stay
    with open(args.stay, "rb") as f:
        stay = pickle.load(f)
    stay = stay[SKIP:]

    # Parse name and value
    tagname, tagvalues = args.tag.split("=")
    tagvalues = set(v.strip() for v in tagvalues.split(","))

    # Retrieve names of NLL columns
    nllnames = get_nllnames(stay, args.concepts)

    # Keep only NLL columns
    nlls = stay[nllnames]
    # skip the ends
    nlls = nlls[args.skip:-args.skip]

    # Compute the rolling mean NLL
    mean_nlls = nlls.mean(axis=1, skipna=True)
    rolling_mean_nll = mean_nlls.rolling(window=args.window).mean()

    # Go through the stay from the end backwards and trace alerts
    tag = stay[tagname]
    j = None
    has_tag = False
    alert = False
    for i in reversed(range(len(nlls))):
        sweetspot = False
        if tag[i] in tagvalues:
            has_tag = True
            j = 0
        if j == args.from_:
            j = None
        if j is not None:
            if j >= args.until_:
                sweetspot = True
            j += 1
        if rolling_mean_nll[i] > args.threshold:
            if sweetspot or not has_tag:
                alert = True
    # print confusions
    print("{:d},{:d},{:d},{:d}".format(
        has_tag and alert, has_tag and not alert,
        not has_tag and alert, not has_tag and not alert))
