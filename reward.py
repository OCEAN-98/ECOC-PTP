import copy
import random
from itertools import product

def newproduct(x, b):
    c = list(product(x, b))
    e = []
    for i in c:
        d = list(i)
        e.append(d)
    return e

# 应该是有两种不同的负回报
# 一个成功一个失败，-1
# 两个都失败 -2

# slot[A] : [是否有占用， 剩余占用时间，调制格式-Modulation]
slot = {'0': [0, 0, 'M'], '1': [0, 0, 'M'], '2': [0, 0, 'M'], '3': [0, 0, 'M'], '4': [0, 0, 'M'], '5': [0, 0, 'M'], '6': [0, 0, 'M'], '7': [0, 0, 'M']} # 这个也得是drl先给一个初始值
timer = 0 # 这个是drl给的
action = [15, 7] # 这个也是drl给的
def action_to_realaction(action):
    x = [0, 1, 2, 3, 4, 5, 6, 7]
    y = ['P', 'Q']
    z = newproduct(x, y)
    real_action = []
    for i in action:
        real_action.append(z[i])
    return real_action

def action_to_reward(action, slot):
    requests = [random.randrange(1, 5, 1), random.randrange(1, 5, 1)]  # [A for how long, B for how long] 这个环境自己产生的,是有真实流量决定，而不是ml来决定
    real_ac = action_to_realaction(action) # [ [which slot, which modulation], [which slot, which modulation] ]
    reward = 0
    a = [x for x in slot.values()]
    slot1 = []
    slot2 = []
    for i in a:
        slot1.append(i[0]) # 这里是记录 这一个timer的slot占用状态
        # 注意 下一个timer的slot的占用状态是 取决于这个action有没有成功的，有可能一个成功，也可能都成功，也可能都失败
    for i in a:
        slot1.append(i[1] / 4)
    for i in a:
        some = 0
        if i[2] == 'M':
            some = 0
        elif i[2] == 'P':
            some = 0.5
        elif i[2] == 'Q':
            some = 1
        slot1.append(some)

    if slot[str(real_ac[0][0])][0] == 1 and slot[str(action[1][0])][0] == 1: # 两个都选择了已经被占用的，所以都不行
        reward = -2
        # slot上的时间还是要都减少一的
        for i in [k for k in slot.keys()]:
            if slot[i][1] > 0:
                slot[i][1] -= 1
                if slot[i][1] == 0:
                    slot[i][0] = 0
                    slot[i][2] = 'M'
        b = [x for x in slot.values()]
        for i in b:
            slot2.append(i[0])
        for i in b:
            slot2.append(i[1]/4)
        for i in b:
            some = 0
            if i[2] == 'M':
                some = 0
            elif i[2] == 'P':
                some = 0.5
            elif i[2] == 'Q':
                some = 1
            slot2.append(some)


    elif slot[str(real_ac[0][0])][0] == 1 and slot[str(real_ac[1][0])][0] == 0: # A不行，B可以
        reward = -1
        # 先看slot怎么变，在才能记录这个slot的占用情况
        slot[str(real_ac[1][0])][0] = 1  # 再把新的时间给加上
        slot[str(real_ac[1][0])][1] = requests[1] # 注意B等一下要经历一个时间步
        slot[str(real_ac[1][0])][2] = real_ac[1][1]
        for i in [k for k in slot.keys()]: # 先把所有的时间都减少一个
            if slot[i][1] > 0:
                slot[i][1] -= 1
                if slot[i][1] == 0:
                    slot[i][0] = 0
                    slot[i][2] = 'M'

        b = [x for x in slot.values()]
        for i in b:
            slot2.append(i[0])  # 这里是记录 这一个timer的slot占用状态
        for i in b:
            slot2.append(i[1]/4)
        for i in b:
            some = 0
            if i[2] == 'M':
                some = 0
            elif i[2] == 'P':
                some = 0.5
            elif i[2] == 'Q':
                some = 1
            slot2.append(some)

    elif slot[str(real_ac[0][0])][0] == 0 and slot[str(real_ac[1][0])][0] == 1: # A可以，B不行
        reward = -1
        # 先看slot怎么变，在才能记录这个slot的占用情况
        slot[str(real_ac[0][0])][0] = 1  # 再把新的时间给加上
        slot[str(real_ac[0][0])][1] = requests[0]  # 注意B等一下要经历一个时间步
        slot[str(real_ac[0][0])][2] = real_ac[0][1]
        for i in [k for k in slot.keys()]:  # 先把所有的时间都减少一个
            if slot[i][1] > 0:
                slot[i][1] -= 1
                if slot[i][1] == 0:
                    slot[i][0] = 0
                    slot[i][2] = 'M'

        b = [x for x in slot.values()]
        for i in b:
            slot2.append(i[0])  # 这里是记录 这一个timer的slot占用状态
        for i in b:
            slot2.append(i[1] / 4)
        for i in b:
            some = 0
            if i[2] == 'M':
                some = 0
            elif i[2] == 'P':
                some = 0.5
            elif i[2] == 'Q':
                some = 1
            slot2.append(some)

    elif action[0] == action[1] and slot[str(action[0])][0] == 0: # 此时原本两个都可以，但是动作一致，最后随机一个行的
        reward = -1
        # 先看slot怎么变，在才能记录这个slot的占用情况
        a = random.choice([0, 1])
        slot[str(real_ac[a][0])][0] = 1  # 再把新的时间给加上
        slot[str(real_ac[a][0])][1] = requests[a]  # 注意B等一下要经历一个时间步
        slot[str(real_ac[a][0])][2] = real_ac[a][1]
        for i in [k for k in slot.keys()]:  # 先把所有的时间都减少一个
            if slot[i][1] > 0:
                slot[i][1] -= 1
                if slot[i][1] == 0:
                    slot[i][0] = 0
                    slot[i][2] = 'M'

        b = [x for x in slot.values()]
        for i in b:
            slot2.append(i[0])  # 这里是记录 这一个timer的slot占用状态
        for i in b:
            slot2.append(i[1] / 4)
        for i in b:
            some = 0
            if i[2] == 'M':
                some = 0
            elif i[2] == 'P':
                some = 0.5
            elif i[2] == 'Q':
                some = 1
            slot2.append(some)


    elif action[0] != action[1] and slot[str(real_ac[0][0])][0] == 0 and slot[str(real_ac[1][0])][0] == 0: # 这是两个都可以发出去都情况
        # 先看slot怎么变，在才能记录这个slot的占用情况
        slot[str(real_ac[0][0])][0] = 1  # 再把新的时间给加上
        slot[str(real_ac[0][0])][1] = requests[0]  # 注意B等一下要经历一个时间步
        slot[str(real_ac[0][0])][2] = real_ac[0][1]
        slot[str(real_ac[1][0])][0] = 1  # 再把新的时间给加上
        slot[str(real_ac[1][0])][1] = requests[1]  # 注意B等一下要经历一个时间步
        slot[str(real_ac[1][0])][2] = real_ac[1][1]

        # 这里就要按照此时的slot安排方式，获得reward了
        b = [x for x in slot.values()]
        check_the_reward_slot = []
        for i in b:
            check_the_reward_slot.append(i[0])
        reward = pick_from_reward_recording(check_the_reward_slot)
        # 这里是要写一个函数，把这个读出来

        for i in [k for k in slot.keys()]:  # 先把所有的时间都减少一个
            if slot[i][1] > 0:
                slot[i][1] -= 1
                if slot[i][1] == 0:
                    slot[i][0] = 0
                    slot[i][2] = 'M'

        for i in b:
            slot2.append(i[0])  # 这里是记录 这一个timer的slot占用状态
        for i in b:
            slot2.append(i[1] / 4)
        for i in b:
            some = 0
            if i[2] == 'M':
                some = 0
            elif i[2] == 'P':
                some = 0.5
            elif i[2] == 'Q':
                some = 1
            slot2.append(some)

    return [requests, reward, slot1, slot2]

#
#
#
#
#
