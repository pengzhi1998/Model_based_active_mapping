"""agent.py
Abstract base class for the various exploration agents.
Must have a plan method, which takes the state (position of the robot), and the current map.
plan should return path to next state and/or actions needed to get there
"""

from __future__ import print_function, absolute_import, division


class Agent:
    """
    Abstract base class for the various exploration agents.
    Must have a plan method, which takes the state (position of the robot), and the current map.
    plan should return path to next state and/or actions needed to get there
    """
    def __init__(self):
        """
        Abstract base class for the various exploration agents.
        Must have a plan method, which takes the state (position of the robot), and the current map.
        plan should return path to next state and/or actions needed to get there
        """
        pass

    def plan(self, state, occupancy_map, debug=False, is_last_plan=False):
        """
        Given the robot position (state) and the map, calculate the path/policy. Returns
        path specifing where the robot should go to act according to the method,
        in the frame of (0, 0) in respect to the map origin
        :param state array(3)[float]: robot position in coordinate space [x, y, theta]
        :param occupancy_map Costmap:  object corresponding to the current map
        :param debug bool: show plots?
        :param is_last_plan bool: whether to trigger last plan logic (i.e plan to start)
        """
        raise NotImplementedError
