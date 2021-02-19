import numpy as np
import matplotlib.pyplot as plt

data_file = "/home/hsuh/Documents/p2t_ais/exp/onions/closed_loop.csv"
data = np.genfromtxt(data_file, delimiter=',')
#data = np.sqrt(data)

plt.figure(figsize=(16,8))
plt.plot(range(data.shape[1]), data[0,:], 'o-', color='royalblue')
plt.plot(range(1,data.shape[1]+1), data[1,:], 'o', color='darkorange')
for i in range(data.shape[1]):
    plt.plot([i, i+1], data[0:2,i], 'k-', linewidth=0.5)
plt.xlabel('iterations')
plt.ylabel('Lyapunov (V)')
plt.ylim([0, 3600])
plt.legend(["actual", "predicted"])
plt.rcParams["font.family"] = "Arial"
plt.show()

plt.figure(figsize=(16,8))
plt.plot(range(data.shape[1]), data[2,:], 'o-', color='royalblue')
plt.plot(range(1,data.shape[1]+1), data[3,:], 'o', color='darkorange')
for i in range(data.shape[1]):
    plt.plot([i, i+1], data[2:4,i], 'k-', linewidth=0.5)
plt.xlabel('iterations')
plt.ylabel('Lyapunov (V)')
plt.ylim([0, 3600])
plt.legend(["actual", "predicted"])
plt.rcParams["font.family"] = "Arial"
plt.show()

plt.figure(figsize=(16,8))
plt.plot(range(data.shape[1]), data[4,:], 'o-', color='royalblue')
plt.plot(range(3,data.shape[1]+3), data[5,:], 'o', color='darkorange')
for i in range(data.shape[1]):
    plt.plot([i, i+3], data[4:6,i], 'k-', linewidth=0.5)
plt.xlabel('iterations')
plt.ylabel('Lyapunov (V)')
plt.ylim([0, 3600])
plt.legend(["actual", "predicted"])
plt.rcParams["font.family"] = "Arial"
plt.show()
