"""frontier_agent.py
Frontier Based Exploration Algorithm, given an occupancy map and the robots pose on the map,
return a path to the "best" frontier.
"""
from __future__ import print_function, absolute_import, division


import cv2
import numpy as np

from functools import partial
from matplotlib import pyplot as plt

from bc_exploration.agents.agent import Agent
from bc_exploration.mapping.costmap import Costmap
from bc_exploration.planners.astar_cpp import oriented_astar, get_astar_angles
from bc_exploration.utilities.util import wrap_angles, which_coords_in_bounds, xy_to_rc, rc_to_xy


def extract_frontiers(occupancy_map, approx=True, approx_iters=2,
                      kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                      debug=False):
    """
    Given a map of free/occupied/unexplored pixels, identify the frontiers.
    This method is a quicker method than the brute force approach,
    and scales pretty well with large maps. Using cv2.dilate for 1 pixel on the unexplored space
    and comparing with the free space, we can isolate the frontiers.

    :param occupancy_map Costmap: object corresponding to the map to extract frontiers from
    :param approx bool: does an approximation of the map before extracting frontiers. (dilate erode, get rid of single
                   pixel unexplored areas creating a large number fo frontiers)
    :param approx_iters int: number of iterations for the dilate erode
    :param kernel array(N, N)[float]: the kernel of which to use to extract frontiers / approx map
    :param debug bool: show debug windows?
    :return List[array(N, 2][float]: list of frontiers, each frontier is a set of coordinates
    """
    # todo regional frontiers
    # extract coordinates of occupied, unexplored, and free coordinates
    occupied_coords = np.argwhere(occupancy_map.data.astype(np.uint8) == Costmap.OCCUPIED)
    unexplored_coords = np.argwhere(occupancy_map.data.astype(np.uint8) == Costmap.UNEXPLORED)
    free_coords = np.argwhere(occupancy_map.data.astype(np.uint8) == Costmap.FREE)

    if free_coords.shape[0] == 0 or unexplored_coords.shape[0] == 0:
        return []

    # create a binary mask of unexplored pixels, letting unexplored pixels = 1
    unexplored_mask = np.zeros_like(occupancy_map.data)
    unexplored_mask[unexplored_coords[:, 0], unexplored_coords[:, 1]] = 1

    # dilate using a 3x3 kernel, effectively increasing
    # the size of the unexplored space by one pixel in all directions
    dilated_unexplored_mask = cv2.dilate(unexplored_mask, kernel=kernel)
    dilated_unexplored_mask[occupied_coords[:, 0], occupied_coords[:, 1]] = 1

    # create a binary mask of the free pixels
    free_mask = np.zeros_like(occupancy_map.data)
    free_mask[free_coords[:, 0], free_coords[:, 1]] = 1

    # can isolate the frontiers using the difference between the masks,
    # and looking for contours
    frontier_mask = ((1 - dilated_unexplored_mask) - free_mask)
    if approx:
        frontier_mask = cv2.dilate(frontier_mask, kernel=kernel, iterations=approx_iters)
        frontier_mask = cv2.erode(frontier_mask, kernel=kernel, iterations=approx_iters)

    # this indexing will work with opencv 2.x 3.x and 4.x
    frontiers_xy_px = cv2.findContours(frontier_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2:][0]
    frontiers = [rc_to_xy(np.array(frontier).squeeze(1)[:, ::-1], occupancy_map) for frontier in frontiers_xy_px]

    if debug:
        frontier_map = np.repeat([occupancy_map.data], repeats=3, axis=0).transpose((1, 2, 0))
        # frontiers = [frontiers[rank] for i, rank in enumerate(frontier_ranks)
        #              if i < self.num_frontiers_considered]
        for frontier in frontiers:
            # if frontier.astype(np.int).tolist() in self.frontier_blacklist:
            #     continue
            frontier_px = xy_to_rc(frontier, occupancy_map).astype(np.int)
            frontier_px = frontier_px[which_coords_in_bounds(frontier_px, occupancy_map.get_shape())]
            frontier_map[frontier_px[:, 0], frontier_px[:, 1]] = [255, 0, 0]

        plt.imshow(frontier_map, cmap='gray', interpolation='nearest')
        plt.show()

    return frontiers


def cleanup_map_for_planning(occupancy_map, kernel, filter_obstacles=False, debug=False):
    """
    We are not allowed to plan in unexplored space, (treated as collision), so what we do is dilate/erode the free
    space on the map to eat up the small little unexplored pixels, allows for quicker planning.
    :param occupancy_map Costmap: object
    :param kernel array(N, N)[float]: kernel to use to cleanup map (dilate/erode)
    :param filter_obstacles bool: whether to filter obstacles with a median filter, potentially cleaning up single
                              pixel noise in the environment.
    :param debug bool: show debug plot?
    :return Costmap: cleaned occupancy map
    """
    occupied_coords = np.argwhere(occupancy_map.data == Costmap.OCCUPIED)
    free_coords = np.argwhere(occupancy_map.data == Costmap.FREE)

    free_mask = np.zeros_like(occupancy_map.data)
    free_mask[free_coords[:, 0], free_coords[:, 1]] = 1
    free_mask = cv2.dilate(free_mask, kernel=kernel, iterations=2)
    free_mask = cv2.erode(free_mask, kernel=kernel, iterations=2)
    new_free_coords = np.argwhere(free_mask == 1)

    if filter_obstacles:
        occupied_mask = np.zeros_like(occupancy_map.data)
        occupied_mask[occupied_coords[:, 0], occupied_coords[:, 1]] = 1
        occupied_mask = cv2.medianBlur(occupied_mask, kernel.shape[0])
        occupied_coords = np.argwhere(occupied_mask == 1)

    cleaned_occupancy_map = occupancy_map.copy()
    cleaned_occupancy_map.data[new_free_coords[:, 0], new_free_coords[:, 1]] = Costmap.FREE
    cleaned_occupancy_map.data[occupied_coords[:, 0], occupied_coords[:, 1]] = Costmap.OCCUPIED

    if debug:
        plt.imshow(cleaned_occupancy_map.data, cmap='gray', interpolation='nearest')
        plt.show()

    return cleaned_occupancy_map


class FrontierAgent(Agent):
    """
    Implementation with similar idea to the paper:

    A Frontier-Based Approach for Autonomous Exploration
    https://pdfs.semanticscholar.org/9afb/8b6ee449e1ddf1268ace8efb4b69578b94f6.pdf
    """
    def __init__(self,
                 footprint,
                 mode='closest',
                 min_frontier_size=10,
                 max_replans=3,
                 planning_resolution=None,
                 planning_epsilon=1.0,
                 planning_delta_scale=1.5):
        """
        Implementation with similar idea to the paper:

        A Frontier-Based Approach for Autonomous Exploration
        https://pdfs.semanticscholar.org/9afb/8b6ee449e1ddf1268ace8efb4b69578b94f6.pdf

        Given a state (robot position) and the current map, compute the frontiers, and using the planner specified,
        plan to the "best" frontier, where "best" is determined by mode. This is achieved by calling the plan method.
        It will return a list of coordinates (path robot must follow to get to the frontier) and using the motion
        model specified in the planner, return the actions needed to follow this path.

        :param footprint CustomFootprint: the footprint of the robot, has a collision checking method
        :param mode str: two word combo joined by '-'. Determines the priority ranking method for the frontiers. The
                     second word in ['closest', 'middle'] determines whether the planner will try to plan to the
                     closest point on the selected frontier or the middle of the frontier.
                     First word is as follows: "best" frontier is chosen by this method
                         closest:  one which is closest (euclidean)
                         largest:  one which is largest (number of adjacent coords in frontier)
                     Some examples would be 'closest-middle' or 'largest-closest'.
        :param min_frontier_size int: the minimum size (number of coordinates in the contour)
                                  for a frontier to be considered for planning
        :param max_replans int: number of replans to give the frontier agent to try different frontiers
                            before returning the best plan to the most promising frontier
        :param planning_resolution float: what resolution to plan on, must be lower than the original map resolution,
                                    allows for much quicker planning, although not guaranteeing a solution even if there
                                    is one, since for tight corners we might need fine movements to fit through.
                                    if it is None, planning is done on the original resolution.
        :param planning_epsilon float: heuristic multiplier for astar planning algorithm
        :param planning_delta_scale float: if the algorithm is unable to plan to any of the frontiers, then how much to
                                     scale the planning goal region by for future replans,
        """

        Agent.__init__(self)
        assert mode in ['closest-closest', 'largest-closest', 'closest-middle', 'largest-middle']

        self._footprint = footprint

        self._mode = mode
        self._min_frontier_size = min_frontier_size

        self._max_replans = max_replans

        self._planning_resolution = planning_resolution
        self._planning_epsilon = planning_epsilon
        self._planning_delta_scale = planning_delta_scale

        self._footprint_masks = None
        self._footprint_outline_coords = None
        self._footprint_mask_radius = None

        self._frontier_blacklist = [[]]
        self._frontiers = []

        self._plan_iteration = 0

        self._planning_angles = get_astar_angles()

    def reset(self):
        """
        Resets member variables, such that the same FrontierAgent object can be used to explore a new environment
        """
        self._footprint_masks = None
        self._footprint_outline_coords = None
        self._footprint_mask_radius = None

        self._frontier_blacklist = [[]]
        self._frontiers = []

        self._plan_iteration = 0

    def get_frontiers(self, compute=False, occupancy_map=None):
        """
        Get the current computed frontiers
        :param compute bool: will compute the latest frontiers for return
        :param occupancy_map Costmap: if compute is true, give the latest map of which to compute from
        :return List(array(N, 2)[float]: list of frontiers, frontier = frontier coordinates
        """
        if compute:
            assert occupancy_map is not None
            frontiers = extract_frontiers(occupancy_map=occupancy_map,
                                          kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
            frontiers = self._filter_frontiers(frontiers, occupancy_map)
            return self._filter_frontiers(frontiers, occupancy_map)

        return self._frontiers

    def get_footprint(self):
        """
        Returns the footprint of the robot used for exploration
        :return CustomFootprint: footprint used for exploration
        """
        return self._footprint

    def _filter_frontiers(self, frontiers, occupancy_map):
        """
        Remove frontiers that we should not consider
        :param frontiers List(array(N, 2)[float]: list of frontiers, corresponding to the output of extract_frontiers
        :param occupancy_map Costmap: corresponding to the current occupancy map
        :return List(array(N, 2)[float]: new_frontiers, the filtered frontiers
        """
        frontier_sizes = np.array([frontier.shape[0] if len(frontier.shape) > 1 else 1 for frontier in frontiers])
        valid_frontier_inds = np.argwhere(frontier_sizes >= self._min_frontier_size)
        return [frontier for i, frontier in enumerate(frontiers)
                if i in valid_frontier_inds and xy_to_rc(frontier, occupancy_map).astype(np.int).tolist()
                not in self._frontier_blacklist]

    def _compute_frontier_ranks(self, state, frontiers):
        """
        Given the state and frontiers, compute a ranking based off the behavioral mode specified
        :param state array(3)[float]: robot pose in coordinate space [x, y, theta (radians)]
        :param frontiers List(array(N, 2)[float]: list of frontiers, computed from extract_frontiers
        :return array(N)[int]: corresponding to the indices of the frontiers list corresponding
                 to a high ranking to low ranking
        """
        # todo do not recompute sizes a bunch of times
        # rank the frontiers based on the mode selected
        if self._mode.split('-')[0] == 'closest':
            frontier_distances = [np.min(np.sqrt(np.sum((state[:2] - frontier) ** 2, axis=1)))
                                  for frontier in frontiers]
            frontier_ranks = np.argsort(frontier_distances)
        elif self._mode.split('-')[0] == 'largest':
            frontier_sizes = [frontier.shape[0] for frontier in frontiers]
            frontier_ranks = np.argsort(frontier_sizes)[::-1]
        else:
            frontier_ranks = []
            assert False and "Mode not supported"

        return frontier_ranks

    def plan(self, state, occupancy_map, debug=False, is_last_plan=False):
        """
        Given the robot position (state) and the occupancy_map, calculate the path/policy to the "best" frontier to
        explore.
        :param state array(3)[float]: robot pose in coordinate space [x, y, theta (radians)]
        :param occupancy_map Costmap: object corresponding to the current map
        :param debug bool: show debug windows?
        :param is_last_plan bool: whether we are done exploring and would like to replan to the starting state
        :return array(N, 3)[float]: array of coordinates of the path
        """
        self._plan_iteration += 1

        # precompute items for collision checking
        if self._footprint_masks is None or self._footprint_mask_radius is None or self._footprint_outline_coords is None:
            self._footprint_masks = self._footprint.get_footprint_masks(occupancy_map.resolution, angles=self._planning_angles)
            self._footprint_outline_coords = self._footprint.get_outline_coords(occupancy_map.resolution, angles=self._planning_angles)
            self._footprint_mask_radius = self._footprint.get_mask_radius(occupancy_map.resolution)

        if self._planning_resolution is not None:
            assert occupancy_map.resolution <= self._planning_resolution
            planning_scale = int(np.round(self._planning_resolution / occupancy_map.resolution))
        else:
            planning_scale = 1

        # this kernel works better than a simple 3,3 ones
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        # get the frontiers on the current map
        exploration_map = occupancy_map.copy()
        frontiers = extract_frontiers(occupancy_map=exploration_map, kernel=kernel)

        frontiers = self._filter_frontiers(frontiers, exploration_map)
        self._frontiers = frontiers

        # rank the frontiers based off the mode
        frontier_ranks = self._compute_frontier_ranks(state, frontiers)

        # todo clean frontier blacklist every N iterations.
        if not len(frontiers):
            self._frontier_blacklist = [[]]

        # if theres no frontiers, there's nothing to plan to, maybe this means we are done
        if not len(frontiers):
            return np.empty((0, 3))

        # clean the occupancy map, eat up small unexplored pixels we dont want to consider a collision
        exploration_map = cleanup_map_for_planning(occupancy_map=exploration_map, kernel=kernel)

        # define our planning function, we use a partial here because most of the inputs are the same
        oriented_astar_partial = partial(oriented_astar,
                                         start=state,
                                         occupancy_map=exploration_map,
                                         footprint_masks=self._footprint_masks,
                                         outline_coords=self._footprint_outline_coords,
                                         obstacle_values=[Costmap.OCCUPIED, Costmap.UNEXPLORED],
                                         planning_scale=planning_scale,
                                         epsilon=self._planning_epsilon)

        num_replans = 0
        frontier_idx = 0
        plan_successful = False
        failsafe_path = None
        tried_blacklist_frontiers = False
        planning_delta = self._footprint_mask_radius
        while not plan_successful and frontier_ranks.shape[0]:
            if frontier_idx > frontier_ranks.shape[0] - 1:
                frontier_idx = 0

                num_replans += 1
                planning_delta *= self._planning_delta_scale

                if num_replans < self._max_replans - 1:
                    print("Not able to plan to frontiers. Increasing planning delta and trying again!")
                else:
                    if failsafe_path is None and not tried_blacklist_frontiers:
                        print("Not able to plan to any frontiers, resetting blacklist")
                        tried_blacklist_frontiers = True
                        self._frontier_blacklist = [[]]
                        continue
                    else:
                        print("Not able to plan to any frontiers, choosing failsafe path")
                        break

            best_frontier = frontiers[frontier_ranks[frontier_idx]]
            assert len(best_frontier.shape) == 2

            if best_frontier.shape[0] == 0:
                continue

            best_frontier_rc_list = xy_to_rc(best_frontier, exploration_map).astype(np.int).tolist()
            if best_frontier_rc_list in self._frontier_blacklist:
                frontier_ranks = np.delete(frontier_ranks, frontier_idx)
                continue

            # based off the mode, we will compute a path to the closest point or the middle of the frontier
            if self._mode.split('-')[1] == 'closest':
                # navigate to the closest point on the best frontier
                desired_coord_ranks = np.argsort(np.sum((state[:2] - best_frontier) ** 2, axis=1))
            elif self._mode.split('-')[1] == 'middle':
                # navigate to the middle of the best frontier
                frontier_mean = np.mean(best_frontier, axis=0)
                desired_coord_ranks = np.argsort(np.sqrt(np.sum((frontier_mean - best_frontier) ** 2, axis=1)))
            else:
                desired_coord_ranks = None
                assert False and "Mode not supported"

            goal = None
            # todo try more angles
            start_frontier_vector = best_frontier[desired_coord_ranks[0]] - state[:2]
            angle_to_frontier = wrap_angles(np.arctan2(start_frontier_vector[0], start_frontier_vector[1]))
            # find a point near the desired coord where our footprint fits
            for _, ind in enumerate(desired_coord_ranks):
                candidate_state = np.concatenate((np.array(best_frontier[ind]).squeeze(), [angle_to_frontier]))
                if not self._footprint.check_for_collision(state=candidate_state, occupancy_map=exploration_map):
                    goal = candidate_state
                    break

            # todo need to integrate goal region at this step. maybe raytrace some poses and try those?
            # if we can't find a pose on the frontier we can plan to, add this frontier to blacklist
            if goal is None:
                self._frontier_blacklist.append(best_frontier_rc_list)
                frontier_ranks = np.delete(frontier_ranks, frontier_idx)
                continue

            if debug:
                frontier_map = np.repeat([exploration_map.data], repeats=3, axis=0).transpose((1, 2, 0)).copy()
                best_frontier = frontiers[frontier_ranks[frontier_idx]]
                best_frontier_vis = xy_to_rc(best_frontier, exploration_map).astype(np.int)
                best_frontier_vis = best_frontier_vis[
                    which_coords_in_bounds(best_frontier_vis, exploration_map.get_shape())]
                frontier_map[best_frontier_vis[:, 0], best_frontier_vis[:, 1]] = [255, 0, 0]

                for _, ind in enumerate(desired_coord_ranks):
                    candidate_state = np.concatenate((np.array(best_frontier[ind]).squeeze(), [angle_to_frontier]))
                    if not self._footprint.check_for_collision(state=candidate_state, occupancy_map=exploration_map):
                        goal = candidate_state
                        break

                goal_vis = xy_to_rc(goal, exploration_map)
                frontier_map[int(goal_vis[0]), int(goal_vis[1])] = [0, 255, 0]
                cv2.circle(frontier_map, tuple(xy_to_rc(state[:2], exploration_map)[::-1].astype(int)),
                           self._footprint.get_radius_in_pixels(exploration_map.resolution), (255, 0, 255), thickness=-1)
                plt.imshow(frontier_map, interpolation='nearest')
                plt.show()

            plan_successful, path = oriented_astar_partial(goal=goal, delta=planning_delta)

            if debug:
                oriented_astar_partial = partial(oriented_astar,
                                                 start=state,
                                                 occupancy_map=exploration_map,
                                                 footprint_masks=self._footprint_masks,
                                                 outline_coords=self._footprint_outline_coords,
                                                 obstacle_values=[Costmap.OCCUPIED, Costmap.UNEXPLORED],
                                                 planning_scale=planning_scale,
                                                 epsilon=self._planning_epsilon)

                plan_successful, path = oriented_astar_partial(goal=goal, delta=planning_delta)

                path_map = np.array(exploration_map.data)
                path_vis = xy_to_rc(path, exploration_map)[:, :2].astype(np.int)
                best_frontier = frontiers[frontier_ranks[frontier_idx]]
                best_frontier_vis = xy_to_rc(best_frontier, exploration_map).astype(np.int)
                best_frontier_vis = best_frontier_vis[
                    which_coords_in_bounds(best_frontier_vis, exploration_map.get_shape())]
                path_map[best_frontier_vis[:, 0], best_frontier_vis[:, 1]] = 75
                path_map[path_vis[:, 0], path_vis[:, 1]] = 175
                plt.imshow(path_map, cmap='gray', interpolation='nearest')
                plt.show()

            if plan_successful and path.shape[0] <= 1:
                plan_successful = False
                self._frontier_blacklist.append(best_frontier_rc_list)
                frontier_ranks = np.delete(frontier_ranks, frontier_idx)
                continue

            for i, pose in enumerate(path):
                if self._footprint.check_for_collision(pose, exploration_map):
                    path = path[:i, :]
                    break

            if failsafe_path is None:
                failsafe_path = path.copy()

            if plan_successful:
                return path

            frontier_idx += 1

        return failsafe_path if failsafe_path is not None else np.array([state])
