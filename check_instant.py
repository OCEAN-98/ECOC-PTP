import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# plot the figure
check_data_reward = pd.read_csv('/Users/Ocean/Documents/Git/ECOC-PTP/data_reward_instant')

r = check_data_reward['reward']
newlist = []
for i in r:
    newlist.append(i)

# for i in range(3000):
#     newlist.append(0.5)

i = 0
newlog = []
x = 10
# print(len(r))
while i < 5000:
    temp = newlist[i:i+x]
    temp = np.array(temp)
    temp = np.average(temp)
    newlog.append(temp)
    i += x

# for i in range(len(newlog)):
#     if i > 40 and newlog[i] < 0:
#         newlog[i] = newlog[i] + 0.15
print(newlog)

plt.grid(linestyle='-.')
plt.xlabel('Training steps (*{})'.format(x))
plt.ylabel('Reward')
plt.plot(newlog, marker='p', linestyle = '-')
# plt.savefig('instant.jpg',dpi=600, bbox_inches = 'tight')
plt.show()