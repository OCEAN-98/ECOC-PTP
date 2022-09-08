import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd
import os
from sklearn.utils import shuffle
from itertools import product
from csv import reader



def generate(file, threshold):
    with open(file, 'r') as csv_file:
        csv_reader = reader(csv_file)
        # Passing the cav_reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)
    # print(list_of_rows)
    shuffle_exe = []
    newlist = []
    for i in range(len(list_of_rows)):
        shuffle_exe.append([])
        for ii in list_of_rows[i]:
            print(ii)
            x = float(ii)
            shuffle_exe[i].append(x)
    #
    # print(shuffle_exe)
    rew = []
    i = 0
    while i < len(shuffle_exe) :
        # print(shuffle_exe[i])
    # for i in shuffle_exe:
        x = shuffle_exe[i][-3:]
        # print(i)

        if x[0] > threshold and x[1] > threshold and x[2] > threshold:
            a = min(x)
            b = max(x)
            for ii in x:
                if ii == a:
                    rew.append(1)
                elif ii == b:
                    rew.append(0)
                else:
                    rew.append(0.5)

        elif x[0] <= threshold or x[1] <= threshold or x[2] <= threshold:
            q = []
            for iiii in x:
                if iiii < threshold:
                    q.append(abs(threshold-iiii))
            a = min(q)
            b = max(q)
            z = []
            for iii in x:
                z.append(abs(threshold-iii))

            for ii in range(len(x)):
                if x[ii] >= threshold:
                    rew.append(-1)
                elif x[ii] < threshold:
                    if z[ii] == a:
                        rew.append(1)
                    elif z[ii] == b:
                        rew.append(0)
                    else:
                        rew.append(0.5)
            # print(x)
            # print(rew)

        aa = shuffle_exe[i][:-3]
    #     print(aa)
        for iiiii in rew:
            aa.append(iiiii)
        newlist.append(aa)
        rew = []
        i += 1
    return newlist

# a = generate('/Users/Ocean/Documents/Git/ECOC-PTP/data_0822_total.csv', 0.003)
# b = generate('/Users/Ocean/Documents/Git/ECOC-PTP/data_0827_total.csv', 0.0045)
# c = generate('/Users/Ocean/Documents/Git/ECOC-PTP/data_0828_total.csv', 0.0047)
# e = generate('/Users/Ocean/Documents/Git/ECOC-PTP/data_1.csv', 0.0049)
# #
import csv
#
def tocsv(list, file):
    for i in range(len(list)):
        f = open(file, 'a', newline='')
        writer = csv.writer(f)
        writer.writerow(list[i])
        f.close()
    # for i in list:
    #     df = pd.DataFrame(i)
    #     df.to_csv(file)

# tocsv(a, '/Users/Ocean/Documents/Git/ECOC-PTP/data_total.csv')
# tocsv(b, '/Users/Ocean/Documents/Git/ECOC-PTP/data_total.csv')
# tocsv(c, '/Users/Ocean/Documents/Git/ECOC-PTP/data_total.csv')
#
# tocsv(a, '/Users/Ocean/Documents/Git/ECOC-PTP/data_total.csv')
# tocsv(b, '/Users/Ocean/Documents/Git/ECOC-PTP/data_total.csv')
# tocsv(c, '/Users/Ocean/Documents/Git/ECOC-PTP/data_total.csv')
#
# tocsv(a, '/Users/Ocean/Documents/Git/ECOC-PTP/data_total.csv')
# tocsv(b, '/Users/Ocean/Documents/Git/ECOC-PTP/data_total.csv')
# tocsv(c, '/Users/Ocean/Documents/Git/ECOC-PTP/data_total.csv')


data = pd.read_csv('/Users/Ocean/Documents/Git/ECOC-PTP/data_back_06.csv')
data = shuffle(data)
data.to_csv('/Users/Ocean/Documents/Git/ECOC-PTP/data_back_07.csv')


# with open('/Users/Ocean/Documents/Git/ECOC-PTP/data_total_1.csv', 'r') as csv_file:
#     csv_reader = reader(csv_file)
#     # Passing the cav_reader object to list() to get a list of lists
#     list_of_rows = list(csv_reader)
# # print(list_of_rows)
#
# shuffle_exe = []
# # newlist = []
# for i in range(len(list_of_rows)):
#     shuffle_exe.append([])
#     for ii in list_of_rows[i]:
#         x = float(ii)
#         shuffle_exe[i].append(x)
#
#
# list = []
# def choworst(a, list):
#     worst_list = []
#     worstcopy = copy.deepcopy(list)
#     worstlist = copy.deepcopy(list)
#     order = [11, 12, 4, 3, 13, 10, 5, 2, 14, 9, 6, 1, 15, 8, 7, 0]
#     while a > 0:
#         # i = 0
#         for i in range(16):
#             if worstlist[order[i]] == 0:
#                 worstlist[order[i]] = 1
#                 break
#         a -= 1
#     if worstcopy != worstlist:
#         worst_list = worstlist
#     return worst_list
#
# withoutreward = []
# for i in range(len(shuffle_exe)):
#     withoutreward.append([])
# for i in range(len(shuffle_exe)):
#     withoutreward[i] = shuffle_exe[i][:16]
#
# for i in range(6):
#     for ii in withoutreward:
#         worst = choworst(i+1, ii)
#         print(str(i))
#         print(ii)
#         print(worst)
#         if worst not in list:
#             list.append(worst)
#
# print(len(list))
# tocsv(list, '/Users/Ocean/Documents/Git/ECOC-PTP/data_1.csv')


#
# e = generate('/Users/Ocean/Documents/Git/ECOC-PTP/data_1.csv', 0.0049)
# tocsv(e, '/Users/Ocean/Documents/Git/ECOC-PTP/data_worst.csv')


# e = generate('/Users/Ocean/Documents/Git/ECOC-PTP/data_0901_total.csv', 0.0019)
# tocsv(e, '/Users/Ocean/Documents/Git/ECOC-PTP/data_worst_extra.csv')


# f = generate('/Users/Ocean/Documents/Git/ECOC-PTP/data_0903_total.csv', 0.00205)
# tocsv(f, '/Users/Ocean/Documents/Git/ECOC-PTP/data_worst_0904.csv')

