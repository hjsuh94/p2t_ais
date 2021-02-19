import matplotlib.pyplot as plt 
import numpy as np
import csv 
import pandas as pd 

my_data = np.sqrt(np.genfromtxt("/home/hsuh/Documents/p2t_ais/exp/onions/prediction.csv", delimiter=','))
plt.figure(figsize=(16,8))
plt.plot(my_data[0,:], 'o-', color='royalblue')
plt.plot(my_data[1,:], 'o-', color='darkorange')
plt.xlabel('iterations')
plt.ylabel('Lyapunov (V)')
plt.ylim([0, 70])
plt.legend(["actual", "predicted"])
plt.show()