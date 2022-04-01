import numpy as np
import cv2
import os

from bc_exploration.envs.grid_world import GridWorld
from bc_exploration.mapping.costmap import Costmap

from rl_mapping.sensors.semantic_sensors import SemanticLidar


class SemanticGridWorld(GridWorld):
    def __init__(self, map_filename,
                 map_resolution,
                 sensor,
                 num_class,
                 footprint,
                 start_state=None,
                 render_size=(500, 500),
                 thicken_obstacles=True,
                 no_collision=True):

        self._no_collision = no_collision

        self._load_semantic_map(map_filename, map_resolution, num_class)

        assert isinstance(sensor, SemanticLidar), "sensor model should be defined as \"SemanticLidar\"."

        GridWorld.__init__(self, '__occ_map__.png',
                           map_resolution,
                           sensor,
                           footprint,
                           start_state,
                           render_size,
                           thicken_obstacles)

        os.remove('__occ_map__.png')

        assert self.semantic_map is not None
        self.sensor.set_map(occupancy_map=self.semantic_map)

    def _load_semantic_map(self, filename, map_resolution, num_class):
        semantic_map_data = cv2.imread(filename)
        assert semantic_map_data is not None, "map file not able to be loaded. Does the file exist?"
        semantic_map_data = cv2.cvtColor(semantic_map_data, cv2.COLOR_RGB2GRAY)
        semantic_map_data = semantic_map_data.astype(np.uint8)

        map_data = semantic_map_data.copy()
        map_data[semantic_map_data != Costmap.FREE] = Costmap.OCCUPIED
        map_data = cv2.cvtColor(map_data, cv2.COLOR_GRAY2RGB)
        cv2.imwrite('__occ_map__.png', map_data)

        # nonsemantic_values = [Costmap.OCCUPIED, Costmap.UNEXPLORED, Costmap.FREE]
        nonsemantic_values = [Costmap.UNEXPLORED, Costmap.FREE]
        map_values = np.unique(semantic_map_data).tolist()
        semantic_values = [v for v in map_values if v not in nonsemantic_values]
        assert num_class == len(semantic_values), "Number of classes is not compatible with the input map."
        for i, sv in enumerate(semantic_values):
            semantic_map_data[semantic_map_data == sv] = i + 1

        self.semantic_map = Costmap(data=semantic_map_data, resolution=map_resolution, origin=[0., 0.])

    def _is_state_valid(self, state, use_inflation=True):
        if self._no_collision:
            return 0 <= state[0] < self.map.get_size()[0] and 0 <= state[1] < self.map.get_size()[1]
        else:
            return 0 <= state[0] < self.map.get_size()[0] and 0 <= state[1] < self.map.get_size()[1] \
                   and not (self.footprint.check_for_collision(state=state, occupancy_map=self.map)
                            if use_inflation else self.footprint_no_inflation.check_for_collision(state=state,
                                                                                                  occupancy_map=self.map))
