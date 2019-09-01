"""Unit tests for code of extract utility.
"""

from intensix.monitor.extract import oname


def test_oname():
    """Tests that oname generates proper output name.
    """
    assert oname("foo", "bar.pkl", 0) == "foo/x-0-bar.pkl"
    assert oname("", "bar.pkl", 1) == "x-1-bar.pkl"
    assert oname("foo", "data/bar.pkl", 2) == "foo/x-2-bar.pkl"
