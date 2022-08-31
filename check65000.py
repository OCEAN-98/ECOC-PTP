from itertools import product
import copy
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

list = []
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

withoutreward = []
for i in range(len(allposs)):
    withoutreward.append([])
for i in range(len(allposs)):
    withoutreward[i] = allposs[i][:16]

for i in range(6):
    for ii in withoutreward:
        worst = choworst(i+1, ii)
        if worst not in list:
            list.append(worst)

print(len(list))
tocsv(list, '/Users/Ocean/Documents/Git/ECOC-PTP/data_all.csv')

