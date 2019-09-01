"""Unit tests for predict.
"""

import pytest
import numpy
from intensix.monitor.predict import *


def test_sync_preds():
    """Tests that sync_pred rearranges predictions correctly.
    """
    preds = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    spreds = sync_preds(preds, 1)
    assert numpy.allclose(spreds, numpy.array([[0, 0, 1, 1],
                                               [1, 2, 3, 4],
                                               [5, 6, 7, 8]]))


def test_sync_nlls():
    """Tests that sync_nlls rearranges nlls correctly.
    """
    nlls = numpy.array([[1, 2], [3, 4]])

    snlls = sync_nlls(nlls, 2)
    # nans cannot be compared, so we check them separately
    nans = numpy.isnan(snlls)
    indices = numpy.logical_not(nans)
    assert numpy.allclose(snlls[indices],
                          numpy.array([[numpy.nan, numpy.nan],
                                       [numpy.nan, numpy.nan],
                                       [1, 2],
                                       [3, 4]])[indices])
