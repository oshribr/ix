"""Unit test for models.
"""

import pytest
import math
import numpy
import torch
from torch.autograd import Variable
from intensix.monitor.models import *


@pytest.fixture
def batch():
    """Returns a batch: time x sample x feature
    """
    # 4 time steps, 2 samples per batch, 2 features
    return torch.Tensor(
        [[[+0.5, -1.0], [-0.5, +0.2]],
         [[+0.1, +0.3], [-0.6, -0.4]],
         [[-0.3, +0.7], [-0.2, -0.1]],
         [[+0.1, +0.2], [+0.6, +0.7]]])


def test_makexy(batch):
    """Tests that input and truth are extracted
    correctly from batch.
    """
    x, y = P(2).makexy(batch, 2)
    assert x.size() == (2, 2, 4), "input shape"
    assert y.size() == (2, 2, 2, 2), "truth shape"
    assert torch.equal(x, torch.Tensor(
        [[[+0.5, -1.0, 0.0, 0.0], [-0.5, +0.2, 0.0, 0.0]],
         [[+0.1, +0.3, 0.0, 0.0], [-0.6, -0.4, 0.0, 0.0]]])), \
        "input data"
    assert torch.equal(y, torch.Tensor(
        [[[[+0.1, +0.3], [-0.3, +0.7]],
          [[-0.6, -0.4], [-0.2, -0.1]]],
         [[[-0.3, +0.7], [+0.1, +0.2]],
          [[-0.2, -0.1], [+0.6, +0.7]]]])), "ground truth"


def test_pred_nlls():
    """Tests that prediction nlls are computed correctly.
    """
    def nll(x, m, s):
        v = s * s
        return math.log(v) + (x - m) * (x - m) / v

    preds = torch.Tensor([[[[0., 1., 2., 1.]]]])
    y = torch.Tensor([[[[0.5, 0.5]]]])

    nlls = P(2).pred_nlls(preds, y, eps=0.)
    assert nlls.size() == (1, 1, 1, 2), "nlls shape"
    assert nlls[0, 0, 0, 0] == pytest.approx(nll(0.5, 0., 2.)), "nlls[0]"
    assert nlls[0, 0, 0, 1] == pytest.approx(nll(0.5, 1., 1.)), "nlls[1]"

    y = torch.Tensor([[[[numpy.nan, numpy.nan]]]])

    print(nlls)
    nlls = P(2).pred_nlls(preds, y).numpy()
    assert numpy.all(numpy.isnan(nlls)), "missing NLLS must be nans"


def test_fill_missing():
    """Tests that missing values are filled correctly.
    """
    # observations with missing (1, 1)
    xi = Variable(torch.Tensor(2, 4))
    xi[:, :2] = 1.
    xi[:, 2:] = 0.
    xi[1, 1] = numpy.nan

    # predictions
    xo = xi.clone()
    xo[:, :2] = 3.
    xo[:, 2:] = 9.

    xf = P(2).fill_missing(xi, xo).data
    assert xf[1, 1] == 3. and xf[1, 3] == 9, "nan replaced"
    assert xf[0, 1] == 1. and xf[0, 2] == 0, "rest stays intact"


def test_P_RNN(batch):
    """Tests that the forward propagation of P_RNN
    basically works.
    """
    # Default number of layers
    model = P_RNN(2, 7, nlayers=2, p=0.5)
    for depth in [1, 3]:
        x, y = model.makexy(batch, depth)
        x = Variable(x)
        xx = model(x, depth)[0]
        assert xx.size() == (batch.size(0) - depth, 2, depth, 4), \
            "depth {}".format(depth)

    # Fix depth for the rest of checks
    depth = 2

    # Check that nlayers=1 works
    model = P_RNN(2, 7, nlayers=1, p=0.5)
    x, y = model.makexy(batch, depth)
    x = Variable(x)
    xx = model(x, depth)[0]
    assert xx.size() == (2, 2, depth, 4), "single layer"

    # Check that works without dropout
    model = P_RNN(2, 7, nlayers=2, p=0)
    x, y = model.makexy(batch, depth)
    x = Variable(x)
    xx = model(x, depth)[0]
    assert xx.size() == (2, 2, depth, 4), "no dropout"


def test_P_GRNN(batch):
    """Tests that the forward propagation of P_GRNN
    basically works.
    """
    # Default number of layers
    model = P_GRNN(2, 7, nlayers=2, p=0.5,
                   encoder_sizes=[16, 8],
                   discriminator_sizes=[16, 8])
    depth = 3
    x, y = model.makexy(batch, depth)
    x = Variable(x)
    xx = model(x, depth)[0]
    assert xx.size() == (1, 2, depth, 4), "GRNN"
