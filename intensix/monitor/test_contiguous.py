"""PyTests for intensix.monitor.contiguous.
"""

from math import nan
import pandas
import pytest
from intensix.monitor.contiguous import *


@pytest.fixture
def df():
    """Returns a regular dataframe.
    """
    return pandas.DataFrame(dict(x=range(10),
                                 y=range(10, 20),
                                 z=range(20, 30)))


@pytest.fixture
def df_with_nans():
    """Returns a dataframe with nans.
    """
    df = pandas.DataFrame(dict(x=range(10),
                               y=range(10, 20),
                               z=range(20, 30)))
    df.loc[0, "x"] = nan
    df.loc[0, "y"] = nan
    df.loc[0, "z"] = nan
    df.loc[1, "x"] = nan
    df.loc[2, "y"] = nan
    df.loc[3, "z"] = nan
    df.loc[7, "y"] = nan
    df.loc[6, "z"] = nan
    return df


def test_boundaries_valid_minrows(df):
    """Tests that minrows must be positive.
    """
    with pytest.raises(AssertionError):
        next(boundaries(df, minrows=0))


def test_boundaries_trivial(df):
    """Tests that boundaries in a frame without
    NaNs span the whole frame.
    """
    assert next(boundaries(df)) == (0, 10)
    assert next(boundaries(df, columns=["x"])) == (0, 10)


def test_boundaries_all_columns(df_with_nans):
    """Tests boundaries on all columns.
    """
    assert next(boundaries(df_with_nans)) == (4, 6)


def test_boundaries_some_columns(df_with_nans):
    """Tests boundaries on some columns.
    """
    assert next(boundaries(df_with_nans, columns=["x"])) == (2, 10)
    assert next(boundaries(df_with_nans, columns=["y"])) == (1, 2)
    assert next(boundaries(df_with_nans, columns=["y", "z"])) == (1, 2)


def test_boundaries_some_columns_minrows(df_with_nans):
    """Tests boundaries on some columns.
    """
    assert next(boundaries(df_with_nans, 2, ["x"])) == (2, 10)
    assert next(boundaries(df_with_nans, 2, ["y"])) == (3, 7)
    with pytest.raises(StopIteration):
        next(boundaries(df_with_nans, 6, ["y", "z"]))


def test_boundaries_multiple_spans(df_with_nans):
    """Tests that multiple spans are returned.
    """
    assert list(boundaries(df_with_nans, 2)) == [(4, 6), (8, 10)]
    assert list(boundaries(df_with_nans, 1, ["x"])) == [(2, 10)]
    assert list(boundaries(df_with_nans, 1, ["y"])) == \
        [(1, 2), (3, 7), (8, 10)]


def test_subframe_all_columns(df):
    """Tests that subframe for all columns works.
    """
    assert subframe(df, 1, 3).equals(
        pandas.DataFrame(dict(x=range(1, 3),
                              y=range(11, 13),
                              z=range(21, 23)),
                         index=range(1, 3)))


def test_subframe_some_columns(df):
    """Tests that subframe for all columns works.
    """
    assert subframe(df, 1, 3, ["x", "z"]).equals(
        pandas.DataFrame(dict(x=range(1, 3),
                              z=range(21, 23)),
                         index=range(1, 3)))


def test_extract(df_with_nans):
    """Test that extract (a thin wrapper) basically works.
    """
    assert list(extract(df_with_nans))
    assert not list(extract(df_with_nans, 6))
