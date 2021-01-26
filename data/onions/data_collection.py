import numpy as np
import cv2
import csv
import os, time
from sim.onions.onion_sim import OnionSim

"""
Script to collect data for pile simulation.
"""

sim = OnionSim()
count = 0
root_dir = "/home/hsuh/Documents/p2t_ais/data/onions/images"
f = open("/home/hsuh/Documents/p2t_ais/data/onions/data.csv", 'w')
writer = csv.writer(f, delimiter=',', quotechar='|')

while(True):
    # refresh every 20 counts.
    if (count % 20 == 0):
        sim.refresh()

    # decrease number of onions every 1000 counts.
    if (count % 1000 == 0) and (count is not 0): 
        sim.change_onion_num(sim.onion_num - 10)

    # exit after collecting enough data.
    if (count % 20000 == 0) and (count is not 0):
        break

    new_filename_i = "{:09d}".format(count) + "i.png"
    new_filename_f = "{:09d}".format(count) + "f.png"

    # save current screenshot. 
    cv2.imwrite(os.path.join(root_dir, new_filename_i), sim.get_current_image())

    # apply a random action.
    u = -0.5 + 1.0 * np.random.rand(4)
    sim.update(u)
    writer.writerow([new_filename_i, u[0], u[1], u[2], u[3], new_filename_f])

    # save resulting screenshot 
    cv2.imwrite(os.path.join(root_dir, new_filename_f), sim.get_current_image())
    count = count + 1    

f.close()