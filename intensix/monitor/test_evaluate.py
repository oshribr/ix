"""Unit tests for evaluate.
"""

import pytest
import pandas
from intensix.monitor.evaluate import *


@pytest.fixture
def stay():
    """Returns a stay dataframe.
    """
    return pandas.DataFrame({"X_nll": range(10),
                             "Y_nll": range(10, 20),
                             "W_nll": range(40, 50),
                             "Z": range(20, 30)})


def test_get_nllnames(stay):
    assert get_nllnames(stay, "X") == ["X_nll"], "single concept"
    assert set(get_nllnames(stay, "X,Y")) == \
        set(["X_nll", "Y_nll"]), "multiple concepts"
    assert set(get_nllnames(stay, None)) == \
        set(["X_nll", "Y_nll", "W_nll"]), "all concepts"
