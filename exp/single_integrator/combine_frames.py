import cv2 
import os
import numpy as np

global_path = "/home/hsuh/Documents/p2t_ais/exp/single_integrator/images"
file_path = os.path.join(global_path, "reward_symmetry_line")

for i in range(500):
    image = cv2.imread(os.path.join(file_path, "image/" + "{:04d}".format(i) + ".png"))
    plot = cv2.imread(os.path.join(file_path, "plot/" + "{:04d}".format(i) + ".png"))    
    plot_resized = cv2.resize(plot, (600, 500))

    concat_image = np.concatenate((image, plot_resized), axis=1)
    cv2.imwrite(os.path.join(file_path, "combined/" + "{:04d}".format(i) + ".png"), concat_image)
    