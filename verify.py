import matplotlib.pyplot as plt
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from csv import reader
import csv

def tocsv(list, file):
    for i in range(len(list)):
        f = open(file, 'a', newline='')
        writer = csv.writer(f)
        writer.writerow(list[i])
        f.close()

def choworst(a, list):
    worst_list = []
    worstcopy = copy.deepcopy(list)
    worstlist = copy.deepcopy(list)
    order = [11, 12, 4, 3, 13, 10, 5, 2, 14, 9, 6, 1, 15, 8, 7, 0]
    while a > 0:
        # i = 0
        for i in range(16):
            if worstlist[order[i]] == 0:
                worstlist[order[i]] = 1
                break
        a -= 1
    if worstcopy != worstlist:
        worst_list = worstlist
    return worst_list

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(17, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, s1): # 21 是剩下两个state的值， 记得在runprocess里调用的时候，输入先变成张量
        # x = torch.cat((s0, s1), 1)
        x = F.relu(self.fc1(s1))
        x = F.relu(self.fc2(x))
        readout = self.fc3(x)  # readout是一个二维向量，分别对应一个动作的预期Q值
        # 由于对应的是Q值，输出的激活函数需要选linear，而不是softmax（全部都小于1）
        return readout

    def init(self):
        self.fc1.weight.data = torch.abs(0.01 * torch.randn(self.fc1.weight.size()))
        self.fc2.weight.data = torch.abs(0.01 * torch.randn(self.fc2.weight.size()))
        self.fc3.weight.data = torch.abs(0.01 * torch.randn(self.fc3.weight.size()))

        self.fc1.bias.data = torch.ones(self.fc1.bias.size()) * 0.01
        self.fc2.bias.data = torch.ones(self.fc2.bias.size()) * 0.01
        self.fc3.bias.data = torch.ones(self.fc3.bias.size()) * 0.01

net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
checkpint = torch.load('/Users/Ocean/Documents/Git/ECOC-PTP/parameters_long')
net.load_state_dict(checkpint['net'])
optimizer.load_state_dict(checkpint['optimizer'])

with open('/Users/Ocean/Documents/Git/ECOC-PTP/data_back.csv', 'r') as csv_file:
    csv_reader = reader(csv_file)
    # Passing the cav_reader object to list() to get a list of lists
    list_of_rows = list(csv_reader)
# print(list_of_rows)

shuffle_exe = []
for i in range(len(list_of_rows)):
    shuffle_exe.append([])
    for ii in list_of_rows[i]:
        x = float(ii)
        shuffle_exe[i].append(x)

# print(len(shuffle_exe))

with open('/Users/Ocean/Documents/Git/ECOC-PTP/data_worst_extra.csv', 'r') as csv_file:
    csv_reader_1 = reader(csv_file)
    # Passing the cav_reader object to list() to get a list of lists
    list_of_rows_1 = list(csv_reader_1)
# print(list_of_rows)


worststate = []
for i in range(len(list_of_rows_1)):
    worststate.append([])
    for ii in list_of_rows_1[i]:
        x = float(ii)
        worststate[i].append(x)


reward = []
timer = 0
s_t = shuffle_exe[0][:17]


while timer < 5000:
    s0 = torch.FloatTensor(s_t).view(-1, 17)
    worst_situaction = choworst(s_t[-1] * 5, s_t[:16])
    # print(worst_situaction)
    x = copy.deepcopy(worst_situaction)
    readout = net(s0)
    readout_t = readout.data.numpy()

    a_t = list(np.zeros(3))
    action_index = np.argmax(readout_t)
    a_t[action_index] = 1
    print(action_index)

    state_and_action = []
    if timer % 1 == 0:
        a = copy.deepcopy(s_t)
        a.append(action_index)
        state_and_action.append(a)
        # tocsv(state_and_action, '/Users/Ocean/Documents/Git/ECOC-PTP/last_three_3.csv')

    r_t = 0

    for i in range(len(worststate)):
        # print(i)
        # print(x)
        # print(x + [action_index + 1])
        # print(i)
        # print(x)
        # if check(x , worststate[i]):
        # print(worststate[i])
        if x == worststate[i][:16]:
            # print('abc')
            r_t = worststate[i][16 + action_index]
            # print(r_t)
            # print(r_t)
        # break
        # print(action_index)
        # print(i)
        # print(r_t)
    # s_t1 = shuffle_exe[timer][:17]

    timer += 1

    s_t = shuffle_exe[timer][:17]


    reward.append(r_t)




xx = 10
abc = 0
newlog = []
# print(len(r))
while abc < 5000:
    temp = reward[abc:abc+xx]
    temp = np.array(temp)
    temp = np.average(temp)
    newlog.append(temp)
    abc += xx


print(newlog)
plt.plot(newlog, '^-',label='DQN')
# plt.plot(maxline, 'g',linewidth = 0.5)
# plt.plot(minline, 'g',linewidth = 0.5)
# plt.plot(aveline, 'r', linewidth = 1)
# plt.fill_between(np.linspace(0, 500, 501), y1=minxx, y2=maxx, where=(maxline > minline), facecolor='y', alpha=0.3)
plt.legend(fontsize=11)
plt.ylim(-0.6, 1)
# plt.xlim(-0.6, 1)
plt.grid(linestyle='-.')
plt.xticks(size = 11)
plt.yticks(size = 11)
plt.xlabel('Verification steps (*{})'.format(10))
plt.ylabel('Reward', size=11)
# plt.savefig('Verifi.jpg',dpi=600, bbox_inches = 'tight')
plt.show()



# check_data_reward = pd.read_csv('/Users/Ocean/Library/Mobile Documents/com~apple~CloudDocs/Documents/TaskOffloading/Code/DQL/Results/data_reward')
#
# r = check_data_reward['reward']
# x = 0
# c = []
# while x < 7000:
#     d = sum(list(r[x: x + 8])) / 8
#     c.append(d)
#     x += 8
#     d = 0
#
# for i in range(len(c)):
#     c[i] = c[i] - 0.08
# for i in range(len(c)):
#     if c[i] < 0:
#         c[i] = 0.1
#
# plt.plot(c)
# plt.ylim(0, 0.8)
# plt.grid(linestyle='-.')
# # plt.xlabel('Iterations')
# # plt.ylabel('Reward')
# plt.show()

