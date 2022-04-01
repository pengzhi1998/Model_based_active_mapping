"""map_transformations.py
map_transformation utility functions
"""

from __future__ import print_function, absolute_import, division

import os
import cv2
import matplotlib.pyplot as plt

from bc_exploration.utilities.paths import get_exploration_dir


def convert_brain_map_to_map(filename, debug=False):
    """
    Given a filename of a brain colored png image of a costmap, convert it to a readable .png map for exploration
    by removing the path, and recoloring the image.
    :param filename str: location of the image to convert
    :param debug bool: show result instead of saving
    """
    image = cv2.imread(filename)
    if image is None:
        assert False and "File not found"
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # path is 150, blue_circle is 29, outside is 70, occupied is 226, free is 0
    gray_image[gray_image == 150] = 255
    gray_image[gray_image == 0] = 255
    gray_image[gray_image == 29] = 255

    gray_image[gray_image == 70] = 0
    gray_image[gray_image == 226] = 0

    if debug:
        plt.imshow(gray_image, cmap='gray', interpolation='nearest')
        plt.show()
    else:
        save_path = get_exploration_dir() + "/maps/" + os.path.basename(filename[:-4]) + ".png"
        cv2.imwrite(save_path, gray_image)
