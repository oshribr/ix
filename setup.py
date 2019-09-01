"""Utilities for learning from monitor time series.

Based on template from
  https://github.com/dtolpin/python-project-skeleton
"""

from setuptools import setup, find_packages
from os import path
import intensix.monitor

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md")) as f:
    long_description = f.read()


setup(
    name="monitor",
    version=intensix.monitor.__version__,

    description="Learning from monitor time series",
    long_description=long_description,
    url="https://intensix.atlassian.net/wiki/"
        "spaces/IN/pages/67010561/"
        "Time+series+prediction+on+monitor+data",

    packages=find_packages(exclude=["doc", "data"]),

    # source code layout
    namespace_packages=["intensix"],

    # Generating the command-line tool
    entry_points={
        "console_scripts": [
            "extract=intensix.monitor.extract:main",
            "prepare=intensix.monitor.prepare:main",
            "train=intensix.monitor.train:main",
            "predict=intensix.monitor.predict:main",
            "changepoints=intensix.monitor.changepoints:main",
            "cp2csv=intensix.monitor.cp2csv:main",
            "evaluate=intensix.monitor.evaluate:main"
        ]
    },

    # author and license
    author="David Tolpin",
    author_email="david@intensix.com",
    license="Proprietary",

    # dependencies, a list of rules
    # Things assumed to be give (e.g. through anaconda):
    # * numpy
    # * scipy
    # * pandas
    install_requires=["PyYAML>=3.12", "pandas"],
    # add links to repositories if modules are not on pypi
    dependency_links=[
    ],

    #  PyTest integration
    setup_requires=["pytest-runner"],
    tests_require=["pytest"]
)
