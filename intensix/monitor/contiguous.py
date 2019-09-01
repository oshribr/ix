"""Extracting contiguous ranges from dataframes.

As per AR-230 (supersedes AR-210), we extract all
contiguous ranges over specified fields (or all fields if not
specified).

We provide here code to run over a data frame and
iterate over the ranges.
"""

import itertools
import numpy


def boundaries(df, minrows=1, columns=None):
    """Iterates over contiguous spans in dataframe
    df at least minrows long. Arguments:
      df - dataframe
      minrows - minimum number of rows in the span
      columns - columns to consider, if None then
                all columns.
    Returns iterator over first (inclusive), last (exclusive).
    """
    assert minrows >= 1, "minrows must be positive"
    if columns is not None:
        # Narrow the frame to specified columns
        df = df[columns]
    first = None
    for i, row in enumerate(
            itertools.chain(
                df.itertuples(index=False),
                # add a guard for a span till the
                # end of the frame
                [[numpy.nan]])):
        if first is None:
            if not numpy.isnan(row).any():
                # Span start
                first = i
        elif numpy.isnan(row).any():
            # Span end
            last = i
            if last - first >= minrows:
                # Found the span
                yield first, last
            first = None


def subframe(df, first, last, columns=None):
    """Extracts a subframe with just the columns, and
    rows between the boundaries. Arguments:
      df - dataframe
      first - first row (inclusive)
      last - last row (exclusive)
      columns - columns to keep.
    Returns the subfame.
    """
    # first extract the rows
    df = df.iloc[first:last]
    # then keep only the relevant columns
    if columns is not None:
        df = df[columns]
    return df


def extract(df, minrows=1, columns=None):
    """Extracts contiguous subframes, a wrapper around
    boundaries and subframe. Returns an iterator over
    subframes.
    """
    for first, last in boundaries(df, minrows, columns):
        yield subframe(df, first, last, columns)
