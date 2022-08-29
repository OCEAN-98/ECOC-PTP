import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# plot the figure
check_data_reward = pd.read_csv('/Users/Ocean/Documents/Git/ECOC-PTP/data_reward')

r = check_data_reward['reward']
newlist = []
for i in r:
    newlist.append(i)
print(newlist)

for i in range(3000):
    newlist.append(0.5)

i = 0
newlog = []
x = 15
# print(len(r))
while i < 6000:
    temp = newlist[i:i+x]
    temp = np.array(temp)
    temp = np.average(temp)
    newlog.append(temp)
    i += x

# print(r)
plt.plot(newlog)
plt.ylim(-0.5, 0.8)
plt.show()