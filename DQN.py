import copy

import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd
import os
from sklearn.utils import shuffle
from itertools import product
import csv


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

def check(list1, list2):
    word1 = ''
    word2 = ''
    for i in list1:
        word1 = word1 + str(i)
    for i in list2:
        word2 = word2 + str(i)
    if word1 in word2 or word2 in word1:
        return True
    else:
        return False

# def newproduct(x, b):
#     c = list(product(x, b))
#     e = []
#     for i in c:
#         d = list(i)
#         e.append(d)
#     return e
# e = newproduct([1, 2, 3], [1, 2])

# data = pd.read_csv('/Users/ocean/Desktop/data.csv')
# data = shuffle(data)
# data.to_csv('/Users/ocean/Desktop/data_shuffle.csv')

# data = pd.read_csv('/Users/Ocean/Documents/Git/ECOC-PTP/data_shuffle.csv')
from csv import reader

# with open('/Users/Ocean/Documents/Git/ECOC-PTP/data_shuffle_1.csv', 'r') as csv_file:
#     csv_reader = reader(csv_file)
#     # Passing the cav_reader object to list() to get a list of lists
#     list_of_rows = list(csv_reader)
# print(list_of_rows)
# shuffle_exe = []
# newlist = []
# for i in range(len(list_of_rows)):
#     shuffle_exe.append([])
#     for ii in list_of_rows[i]:
#         x = float(ii)
#         shuffle_exe[i].append(x)
# #
# # print(shuffle_exe)
# rew = []
# i = 0
# while i < len(shuffle_exe) :
#     # print(shuffle_exe[i])
# # for i in shuffle_exe:
#     x = shuffle_exe[i][-3:]
#     # print(i)
#
#     if x[0] > 0.0030 and x[1] > 0.0030 and x[2] > 0.0030:
#         a = min(x)
#         b = max(x)
#         for ii in x:
#             if ii == a:
#                 rew.append(1)
#             elif ii == b:
#                 rew.append(0)
#             else:
#                 rew.append(0.5)
#
#     elif x[0] <= 0.0030 or x[1] <= 0.0030 or x[2] <= 0.0030:
#         q = []
#         for iiii in x:
#             if iiii < 0.003:
#                 q.append(abs(0.003-iiii))
#         a = min(q)
#         b = max(q)
#         z = []
#         for iii in x:
#             z.append(abs(0.003-iii))
#
#         for ii in range(len(x)):
#             if x[ii] >= 0.003:
#                 rew.append(-1)
#             elif x[ii] < 0.003:
#                 if z[ii] == a:
#                     rew.append(1)
#                 elif z[ii] == b:
#                     rew.append(0)
#                 else:
#                     rew.append(0.5)
#         # print(x)
#         # print(rew)
#
#     aa = shuffle_exe[i][:-3]
# #     print(aa)
#     for iiiii in rew:
#         aa.append(iiiii)
#     newlist.append(aa)
#     rew = []
#     i += 1
# # print(shuffle_exe[-3:])
# # print(newlist[-3:])
# # print(len(shuffle_exe))
# # print(len(newlist))



#
with open('/Users/Ocean/Documents/Git/ECOC-PTP/data_back_07.csv', 'r') as csv_file:
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

print(shuffle_exe)


with open('/Users/Ocean/Documents/Git/ECOC-PTP/data_0903_total.csv', 'r') as csv_file_1:
    csv_reader_1 = reader(csv_file_1)
    # Passing the cav_reader object to list() to get a list of lists
    list_of_rows_1 = list(csv_reader_1)

# print(list_of_rows)

worst_file = []
for i in range(len(list_of_rows_1)):
    worst_file.append([])
    for ii in list_of_rows_1[i]:
        # print(ii)
        x = float(ii)
        worst_file[i].append(x)

worststate = []
for i in worst_file:
    worststate.append(i[:16])

print(worststate)

def tocsv(list, file):
    for i in range(len(list)):
        f = open(file, 'a', newline='')
        writer = csv.writer(f)
        writer.writerow(list[i])
        f.close()


def matrix_to_vector(matrix):
    list = []
    for i in matrix:
        for ii in matrix[i]:
            list.append(matrix[i][ii])
    return list

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(17, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 10)

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


import numpy as np
import torch
import pandas as pd

ACTIONS = 10 # action个数， 贪婪算法
GAMMA = 0.8 # 衰减率
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.001
REPLAY_MOMERY = 2000 #1000
BATCH = 200 #50
OBSERVE = 2000 #1000
EXPLORE = 5000 #6000
TRAIN = 4000 #3000

net = Net() # 神经网络
net.init()
# net.cuda()
criterion = nn.MSELoss() #.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

# Offload = stepgo(301, 6)
D = []
# newlist = [] # 10000行数据

s_t = shuffle_exe[0][:16]

# print(worst_file[0])


epsilon = INITIAL_EPSILON
timer = 1
state = ''
loss_value = []
reward = []
time_slot_l = []
time_slot_r = []
time = shuffle_exe[0][-1] * 5
length = 6
# print(s_t)
# if [1.0, 1.0, 1, 1.0, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1, 0.0, 1.0] in worst_file:
#     print(True)
# else:
#     print(False)


while timer < (OBSERVE + EXPLORE + TRAIN): # 把输出换成numpy格式，是一个很长的list，注意有两个卷积网，需要分几个S出来
    # print(s_t)

        # random.randrange(1, length)

    # print(s_t)
    worst_situaction = choworst(time, s_t)
    # print(worst_situaction)
    x = copy.deepcopy(worst_situaction)
    # x[-1] = x[-1] / 5
    # if x in worststate:
    #     print(True)
    s_t_copy = copy.deepcopy(s_t)
    s_t_copy.append(time / (length - 1))
    s = torch.FloatTensor(s_t_copy).view(-1, 17)
    # print(s)
    readout = net(s)
    # readout = readout.cpu() # readout是一个二维向量，分别是Q值和对应的action选择
    readout_t = readout.data.numpy()

    a_t = list(np.zeros(10))
    action_index = 0
    if random.random() <= epsilon:
        action_index = random.randrange(ACTIONS) # 随机选择行动
    else:
        action_index = np.argmax(readout_t) # 最大Q值选择行动
    a_t[action_index] = 1

    r_t = 0

    check_debug_1 = 0
    check_debug_2 = 0
    # print(x)
    # print(len(worststate))
    for i in range(len(worststate)):
        # print(i)
        # print(x)
        # print(x + [action_index + 1])
        # print(i)
        # print(x)
        # if check(x , worststate[i]):
        if x == worststate[i]:
            # print('abc')
            r_t = worst_file[i][16 + action_index]
            check_debug_1 = worststate[i]
            check_debug_2 = x

    # if r_t == 0:
        # print(check_debug_1)
        # print(check_debug_2)
            # print(r_t)
            # print(r_t)
        # break
            # print(action_index)
            # print(i)
            # print(r_t)

    if epsilon > FINAL_EPSILON and timer > OBSERVE:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    s_t1 = shuffle_exe[timer][:16]
    time_1 = random.randrange(1, length)/5
    s_t1_copy = copy.deepcopy(s_t1)
    s_t1_copy.append(time_1)

    D.append([s_t_copy, a_t, r_t, s_t1_copy])
    if len(D) > REPLAY_MOMERY:
        D = D[1:]

    if timer > OBSERVE:
        starter = random.randrange(0, REPLAY_MOMERY - BATCH)
        minibatch = D[starter:(starter + BATCH)]
        # print(minibatch)
        optimizer.zero_grad()

        s0_j_batch = list([d[0] for d in minibatch])
        # print(s0_j_batch)
        a_batch = list([d[1] for d in minibatch])
        r_batch = list([d[2] for d in minibatch])

        s0_j1_batch = list([d[3] for d in minibatch])

        s1 = torch.FloatTensor(s0_j1_batch).view(-1, 17)

        readout1 = net(s1)
        # # readout1 = readout1.cpu()
        readout_j1_batch = readout1.data.numpy()
        #
        y_batch = []
        for i in range(0, len(minibatch)):
            y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
        y = torch.from_numpy(np.array(y_batch, dtype=float))
        a = torch.from_numpy(np.array(a_batch, dtype=float))

        s0 = torch.FloatTensor(s0_j_batch).view(-1, 17)

        readout0 = net(s0)
        readout_action = readout0.mul(a).sum(1)
        loss = criterion(readout_action, y)
        # print(loss)
        loss.backward()
        optimizer.step()
        if timer % 10 == 0:
            loss = loss.detach().numpy()
            print('loss', loss)
            loss_value.append(loss)
            time_slot_l.append(timer)

    timer += 1
    # print(Offload.AfterDownlog)
    # print(Offload.AfterDownpath)
    # if Vect < 7:
    #     Vect += 1
    # else:
    #     Vect = 0

    s_t = shuffle_exe[timer][:16]
    time = shuffle_exe[timer][-1] * 5

    gotta = {'net': net.state_dict(), 'optimizer':optimizer.state_dict()}

    if timer % 1 == 0:  # == (OBSERVE + EXPLORE + TRAIN):
        torch.save(gotta, '/Users/Ocean/Documents/Git/ECOC-PTP/parameters_long')

    if timer <= OBSERVE:
        state = 'observe'
    elif timer > OBSERVE and timer <= OBSERVE + EXPLORE:
        state = 'explore'  # 开始训练
    else:
        state = 'train'  # 贪婪策略已经结束

    if timer % 1 == 0:
        sss = 'time_step {}/ state {}/ Epsilon {:.2f}/ action {}/ reward {}/ Q_MAX {:e}/'.format(
            timer, state, epsilon, action_index, r_t, np.max(readout_t)
        )
        print(sss)
        f = open('/Users/Ocean/Documents/Git/ECOC-PTP/log_file_long.txt', 'a')
        f.write(sss + '\n')
        f.close()

    if timer % 1 == 0:
        time_slot_r.append(timer)
        reward.append(r_t)

    state_and_action = []
    if timer % 1 == 0:
        a = copy.deepcopy(s_t)
        a.append(action_index)
        state_and_action.append(a)
        tocsv(state_and_action, '/Users/Ocean/Documents/Git/ECOC-PTP/last_three_2.csv')

    if timer % 1 == 0:
        data_loss = {'loss': loss_value,
                     'time_l': time_slot_l}
        data_reward = {'reward': reward,
                       'time_r': time_slot_r}

        data_loss = pd.DataFrame(data_loss)
        data_loss.to_csv('/Users/Ocean/Documents/Git/ECOC-PTP/data_loss_long', index=False)
        data_reward = pd.DataFrame(data_reward)
        data_reward.to_csv('/Users/Ocean/Documents/Git/ECOC-PTP/data_reward_long')



# #
# # check_data_loss = pd.read_csv('/Users/Ocean/Library/Mobile Documents/com~apple~CloudDocs/Documents/TaskOffloading/Code/DQL/Results/data_loss')
# # check_data_reward = pd.read_csv('/Users/Ocean/Library/Mobile Documents/com~apple~CloudDocs/Documents/TaskOffloading/Code/DQL/Results/data_reward')
# # import matplotlib.pyplot as plt
# #
# # l = check_data_loss['loss']
# # r = check_data_reward['reward']
# # t_l = check_data_loss['time_l']
# # t_r = check_data_reward['time_r']
# # plt.plot(t_l, l)
# # plt.plot(t_r, r)
#
#
