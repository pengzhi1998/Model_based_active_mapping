import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
print(cur_path)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))
print(os.path.abspath("."))

from bc_exploration.mapping.costmap import Costmap

from rl_mapping.agents.gradient_agent import create_gradient_agent_from_params
from rl_mapping.sensors.semantic_sensors import SemanticLidar
from rl_mapping.envs.semantic_grid_world import SemanticGridWorld
from rl_mapping.mapping.mapper_kf import KFMapper
from rl_mapping.utilities.utils import visualize, visualize_traj, visualize_info

# os.chdir(cur_path)
print(os.path.abspath("."))

def run_multiclass_gradient_exploration(map_filename, params_filename, sensor_range, angular_range, std,
                                        map_resolution, num_class, start_state=None, render=True, render_interval=1,
                                        render_size_scale=1.7, render_wait_for_key=True,
                                        max_exploration_iterations=None, completion_check_interval=5,
                                        completion_percentage=0.8):

    # define agent
    gradient_agent = create_gradient_agent_from_params(params_filename=params_filename, sensor_range=sensor_range,
                                                       angular_range=angular_range, sigma2=std**2)
    footprint = gradient_agent.get_footprint()

    # pick a sensor
    sensor = SemanticLidar(sensor_range=sensor_range,
                           angular_range=angular_range,
                           angular_resolution=0.5 * np.pi / 180,
                           map_resolution=map_resolution,
                           num_classes=num_class,
                           aerial_view=True)

    # setup grid world environment
    env = SemanticGridWorld(map_filename=map_filename,
                            map_resolution=map_resolution,
                            sensor=sensor,
                            num_class=num_class,
                            footprint=footprint,
                            start_state=start_state,
                            no_collision=True)

    render_size = (np.array(env.get_map_shape()[::-1]) * render_size_scale).astype(np.int)

    # setup semantic mapper, we assume the measurements are very accurate,
    # thus one scan should be enough to fill the map
    padding = 0.
    map_shape = np.array(env.get_map_shape()) + int(2. * padding // map_resolution)
    initial_map = Costmap(data=Costmap.UNEXPLORED * np.ones(map_shape, dtype=np.uint8),
                          resolution=env.get_map_resolution(),
                          origin=[-padding - env.start_state[0], -padding - env.start_state[1]])

    mapper = KFMapper(initial_map=initial_map, sigma2=std**2)

    # reset the environment to the start state, map the first observations
    pose, obs = env.reset()

    occ_map = mapper.update(state=pose, obs=obs)

    if render:
        ppp = 0
        # visualize(occupancy_map=semantic_map, state=pose, footprint=footprint,
        #           start_state=start_state, scan_angles=scan_angles, scan_ranges=scan_ranges,
        #           render_size=render_size, wait_key=0 if render_wait_for_key else 1)
        visualize(state=pose, semantic_map=occ_map, num_class=1, render_size=render_size, wait_key=0, save_file=str(ppp)+".png")
        visualize_info(info_map=mapper.get_distrib_map(), render_size=render_size, wait_key=0, save_file=str(ppp) + "_info.png")
        ppp += 1

    iteration = 0
    is_last_plan = False
    was_successful = True
    percentage_explored = 0.0
    # map_entropy = [compute_map_entropy(mapper.get_probability_map())]
    percentage_explored_list = [0]
    # dist = [0]
    reward = []
    while True:
        # if iteration % completion_check_interval == 0:
        #     percentage_explored = env.compare_maps(mapper.get_occupancy_map())
        #     percentage_explored_list.append(percentage_explored)
        #     # if percentage_explored >= completion_percentage:
        #     #     is_last_plan = True

        if max_exploration_iterations is not None and iteration > max_exploration_iterations:
            was_successful = False
            is_last_plan = True

        # using the current map, make an action plan
        path, init_path, opt_path, new_reward = gradient_agent.plan(state=pose, distrib_map=mapper.get_distrib_map(),
                                                                    face_unknown=True)

        # if we get empty lists for policy/path, that means that the agent was
        # not able to find a path/frontier to plan to.
        if not path.shape[0]:
            print("No path found! Either the map is 100% explored, or bad start state, or there is a bug!")
            break

        # if we have a policy, follow it until the end. update the map sparsely (speeds it up)
        for j, desired_state in enumerate(path):
            pose, obs = env.step(desired_state)
            occ_map = mapper.update(state=pose, obs=obs)
            reward.append(mapper.get_info_reward())

            # put the current laserscan on the map before planning
            # occupied_coords = scan_to_points(scan_angles + pose[2], scan_ranges) + pose[:2]
            # occupied_coords = xy_to_rc(occupied_coords, occupancy_map).astype(np.int)
            # occupied_coords = occupied_coords[which_coords_in_bounds(occupied_coords, occupancy_map.get_shape())]
            # occupancy_map.data[occupied_coords[:, 0], occupied_coords[:, 1]] = Costmap.OCCUPIED

            # shows a live visualization of the exploration process if render is set to true
            if render and j % render_interval == 0:
                # visualize(occupancy_map=semantic_map, state=pose, footprint=footprint,
                #           start_state=start_state, scan_angles=scan_angles, scan_ranges=scan_ranges,
                #           path=path, render_size=render_size,
                #           frontiers=frontier_agent.get_frontiers(compute=True, occupancy_map=semantic_map), wait_key=1,
                #           path_idx=j)
                visualize_traj(state=pose, old_traj=init_path, new_traj=opt_path, semantic_map=occ_map,
                               num_class=1, render_size=render_size, wait_key=0, save_file=str(ppp)+".png")
                visualize_info(info_map=mapper.get_distrib_map(), render_size=render_size, wait_key=0,
                               save_file=str(ppp) + "_info.png")
                ppp += 1

        # map_entropy.append(compute_map_entropy(mapper.get_probability_map()))
        # dist.append(np.sum(np.linalg.norm(path[1:, :2] - path[:-1, :2], axis=1)))

        if is_last_plan:
            break

        iteration += 1

    if render:
        cv2.waitKey(0)

    # percentage_explored_list = np.array(percentage_explored_list)
    # map_entropy = np.array(map_entropy)
    # dist = np.array(dist)
    # perf = np.hstack((map_entropy[:, None], dist[:, None]))

    reward = np.asarray(reward)
    plt.figure()
    plt.plot(np.asarray(reward))
    plt.show()

    np.save('map_6_entropy_enhanced', reward)

    return occ_map#, perf, percentage_explored_list, percentage_explored, iteration, was_successful


def main():
    """
    Main Function
    """
    print(os.path.abspath(os.path.join("", os.pardir)))
    # np.random.seed(3)
    occ_map = \
        run_multiclass_gradient_exploration(map_filename=os.path.join(os.path.abspath(os.path.join("", os.pardir)), "maps/map10_converted.png"),
                                            params_filename=os.path.join(os.path.abspath(os.path.join("", os.pardir)), "params/params.yaml"),
                                            map_resolution=0.03,
                                            start_state=None,
                                            sensor_range=1.5,
                                            angular_range=np.pi*2,
                                            std=0.01,
                                            num_class=1,
                                            completion_percentage=0.97,
                                            max_exploration_iterations=35,
                                            render=True,
                                            render_interval=100)

    # print("Map", "{:.2f}".format(percent_explored * 100), "\b% explored!",
    #       "This is " + str(iterations_taken + 1) + " iterations!")


if __name__ == '__main__':
    main()