import numpy as np
import matplotlib.pyplot as plt

data_file = "/home/hsuh/Documents/p2t_ais/exp/onions/online_rl.csv"
data = np.genfromtxt(data_file, delimiter=',')

plt.figure(figsize=(16,8))
actual_idx = 494
predicted_idx = actual_idx + 1
plt.plot(range(data.shape[1]), data[actual_idx,:], 'o-', color='royalblue')
plt.plot(range(1,data.shape[1]+1), data[predicted_idx,:], 'o', color='darkorange')

for i in range(data.shape[1]):
    plt.plot([i, i+1], data[actual_idx:predicted_idx+1,i], 'k-', linewidth=0.5)
plt.xlabel('iterations')
plt.ylabel('Lyapunov (V)')
#plt.ylim([0, 60])
plt.legend(["actual", "predicted"])
plt.rcParams["font.family"] = "Arial"
plt.show()
