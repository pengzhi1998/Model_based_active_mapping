import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    map_data = cv2.imread('map10.png')
    map_data = cv2.cvtColor(map_data, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    map_data = (cv2.erode(map_data, kernel, iterations=3) > 150).astype(np.uint8) * 255

    cv2.imwrite('map10_converted.png', map_data)

    plt.figure()
    plt.imshow(map_data > 150, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()