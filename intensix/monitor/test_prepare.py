"""Unit tests for the code of prepare utility.
"""

from intensix.monitor.prepare import standardize, reshape
import numpy


def test_standardize():
    """Tests that standardize indeed standardizes
    matrix columnwise.
    """
    x = numpy.random.randn(13, 7) * 4 + 1
    y, scalex = standardize(x)
    assert numpy.allclose(y.mean(), 0)
    assert numpy.allclose(y.std(), 1)
    assert numpy.allclose(y * scalex[1, :] + scalex[0, :], x)


def test_reshape():
    """Tests that the dataset is properly reshaped.
    """
    y = reshape(numpy.array(range(12)).reshape((-1, 2)), 3)
    assert y.shape == (2, 3, 2)
