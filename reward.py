import copy
import random

# 应该是有两种不同的负回报
# 一个成功一个失败，-1
# 两个都失败 -2

# slot[A] : [是否有占用， 剩余占用时间，调制格式-Modulation]
slot = {'0': [0, 0, 'M'], '1': [0, 0, 'M'], '2': [0, 0, 'M'], '3': [0, 0, 'M'], '4': [0, 0, 'M'], '5': [0, 0, 'M'], '6': [0, 0, 'M'], '7': [0, 0, 'M']} # 这个也得是drl先给一个初始值
timer = 0 # 这个是drl给的
action = [7, 7] # 这个也是drl给的
def action_to_reward(action, slot):
    requests = [random.randrange(1, 5, 1), random.randrange(1, 5, 1)] # 这个环境自己产生的
    reward = 0
    a = [x for x in slot.values()]
    slot1 = []
    for i in a:
        slot1.append(i[0]) # 这里是记录 这一个timer的slot占用状态
        # 注意 下一个timer的slot的占用状态是 取决于这个action有没有成功的，有可能一个成功，也可能都成功，也可能都失败
    # if action[0] == action[1]:
    #
    #     reward = -1
    # else:

    if slot[str(action[0])][0] == 1 and slot[str(action[1])][0] == 1: # 两个都选择了已经被占用的，所以都不行
        reward = -2
    elif slot[str(action[0])][0] == 1 and slot[str(action[1])][0] == 0: # A不行，B可以
        reward = -1
    elif slot[str(action[0])][0] == 0 and slot[str(action[1])][0] == 1: # A可以，B不行
        reward = -1
    elif action[0] == action[1] and slot[str(action[0])][0] == 0: # 此时原本两个都可以，但是动作一致



    return [requests, reward, slot]








how_long_to_use = random.randrange(0, 4)


# def action_to_realaction(action):
#     activated_node = []
#     PP1list = ['Node1', 'Node3', 'Node5', 'Node6', 'Node8', 'Node10']
#     PP2list = ['Node2', 'Node4', 'Node7', 'Node9']
#     PP3list = ['Node11', 'Node12', 'Node13', 'Node14']
#
#     PP1combination = [] # 最终长度 64
#     PP2combination = [] # 最终长度 16
#     PP3combination = [] # 最终长度 16
#     for i in range(len(PP1list)):
#         PP1combination.extend(combine(PP1list, i))
#     PP1combination.append(('Node1', 'Node3', 'Node5', 'Node6', 'Node8', 'Node10'))
#
#     for i in range(len(PP2list)):
#         PP2combination.extend(combine(PP2list, i))
#     PP2combination.append(('Node2', 'Node4', 'Node7', 'Node9'))
#
#     for i in range(len(PP3list)):
#         PP3combination.extend(combine(PP3list, i))
#     PP3combination.append(('Node11', 'Node12', 'Node13', 'Node14'))
#
#     real_action_PP1 = PP1combination[action[0]]
#     real_action_PP2 = PP2combination[action[1]]
#     real_action_PP3 = PP3combination[action[2]]
#     for i in real_action_PP1:
#         activated_node.append(i)
#     for i in real_action_PP2:
#         activated_node.append(i)
#     for i in real_action_PP3:
#         activated_node.append(i)
#     return activated_node