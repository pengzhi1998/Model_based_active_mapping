"""paths.py
path based utility functions for exploration
"""

from __future__ import print_function, absolute_import, division

import os


def get_exploration_dir():
    """
    Get the exploration root dir
    :return str: exploration root dir
    """
    return os.path.dirname(os.path.dirname(__file__))


def get_maps_dir():
    """
    Get the exploration/maps dir
    :return str: exploration/maps dir
    """
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "maps")
