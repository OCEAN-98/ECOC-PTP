import random


# slot[A] : [是否有占用， 剩余占用时间，调制格式]
slot = {'A': [0, 0, 'M'], 'B': [0, 0, 'M'], 'C': [0, 0, 'M'], 'D': [0, 0, 'M'], 'E': [0, 0, 'M'], 'F': [0, 0, 'M'], 'G': [0, 0, 'M'], 'H': [0, 0, 'M']}
timer = 0
requests = {}

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