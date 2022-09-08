import csv
from csv import reader
import copy

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

with open('/Users/Ocean/Documents/Git/ECOC-PTP/last_three_3.csv', 'r') as csv_file:
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

with open('/Users/Ocean/Documents/Git/ECOC-PTP/data_0901_total.csv', 'r') as csv_file:
    csv_reader_2 = reader(csv_file)
    # Passing the cav_reader object to list() to get a list of lists
    list_of_rows_2 = list(csv_reader_2)
# print(list_of_rows)

allchecklist = []
for i in range(len(list_of_rows_2)):
    allchecklist.append([])
    for ii in list_of_rows_2[i]:
        x = float(ii)
        allchecklist[i].append(x)

with open('/Users/Ocean/Documents/Git/ECOC-PTP/data_worst_extra.csv', 'r') as csv_file:
    csv_reader_3 = reader(csv_file)
    # Passing the cav_reader object to list() to get a list of lists
    list_of_rows_3 = list(csv_reader_3)
# print(list_of_rows)

changedreward = []
for i in range(len(list_of_rows_3)):
    changedreward.append([])
    for ii in list_of_rows_3[i]:
        x = float(ii)
        changedreward[i].append(x)

# print(worststate)

for i in worststate:
    i[-2] = i[-2] * 5

print(worststate)

# # print(worststate)
# for i in worststate:
#     for ii in range(len(allchecklist)):
#         print(choworst(i[16], i[:16]))
#         # print(allchecklist[ii][:16])
#         # print(i[1])
#         if choworst(i[17], i[:16]) == allchecklist[ii][:16]:
#             i.append(allchecklist[ii][16])
#             i.append(allchecklist[ii][17])
#             i.append(allchecklist[ii][18])
#             i.append(changedreward[ii][16])
#             i.append(changedreward[ii][17])
#             i.append(changedreward[ii][18])
#
# tocsv(worststate, '/Users/Ocean/Documents/Git/ECOC-PTP/last_three_3_1.csv')