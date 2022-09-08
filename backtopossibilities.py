import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd
import os
from sklearn.utils import shuffle
from itertools import product
from csv import reader
import csv


def newproduct(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
    xx = list(product(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p))
    xxxx = []
    for z in xx:
        xxx = list(z)
        xxxx.append(xxx)
    return xxxx

def tocsv(list, file):
    for i in range(len(list)):
        f = open(file, 'a', newline='')
        writer = csv.writer(f)
        writer.writerow(list[i])
        f.close()

a = [0, 1]
b = [0, 1]
c = [0, 1]
d = [0, 1]
e = [0, 1]
f = [0, 1]
g = [0, 1]
h = [0, 1]
i = [0, 1]
j = [0, 1]
k = [0, 1]
l = [0, 1]
m = [0, 1]
n = [0, 1]
o = [0, 1]
p = [0, 1]
# r = [1/5, 2/5, 3/5, 4/5, 5/5]

x = product(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
xx = list(x)
allposs = []
for i in range(len(xx)):
    allposs.append([])
for i in range(len(xx)):
    for ii in xx[i]:
        allposs[i].append(ii)


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


# with open('/Users/Ocean/Documents/Git/ECOC-PTP/data_all.csv', 'r') as csv_file:
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


with open('/Users/Ocean/Documents/Git/ECOC-PTP/data_0903_total.csv', 'r') as csv_file:
    csv_reader_1 = reader(csv_file)
    # Passing the cav_reader object to list() to get a list of lists
    list_of_rows_1 = list(csv_reader_1)
# print(list_of_rows)

shuffle_exe_1 = []
# newlist = []
for i in range(len(list_of_rows_1)):
    shuffle_exe_1.append([])
    for ii in list_of_rows_1[i]:
        x = float(ii)
        shuffle_exe_1[i].append(x)

print(len(shuffle_exe_1))

withoutreward = []
for i in range(len(shuffle_exe_1)):
    withoutreward.append([])
for i in range(len(shuffle_exe_1)):
    withoutreward[i] = shuffle_exe_1[i][:16]

print(len(withoutreward))
print(len(allposs))

backpass = []
aa = 0
for i in allposs:
    # print(i)
    for a in range(1, 6):
        # print(choworst(a, i))
        if choworst(a, i) in withoutreward and i not in backpass:
            # print(i)
            x = copy.deepcopy(i)
            x.append(a/5)
            backpass.append(x)

print(len(backpass))
tocsv(backpass, '/Users/Ocean/Documents/Git/ECOC-PTP/data_back_05.csv')




# check = []
# for i in backpass:
#     for ii in range(1, 6):
#         if choworst(ii, i) in withoutreward and choworst(ii, i) not in check:
#             check.append(choworst(ii, i))
#
# print(len(check))





