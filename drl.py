import os
import random
import pandas as pd
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from reward import *
import matplotlib.pyplot as plt


class MultiAgentReplayBuffer:  # 每个actor有partial state； 而中央控制器有全部state
    def __init__(self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.actor_dims = actor_dims
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.state_memory = np.zeros((self.mem_size, critic_dims))  # 比如缓存10000个，10000行[]的矩阵
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))  #
        self.reward_memory = np.zeros((self.mem_size, n_agents))  # 10000个回报
        self.action_memory = []  # 8 * 50000 * 20
        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []  # 用来记录所有agent的state
        self.actor_new_state_memory = []  # 用来记录
        # self.actor_action_memory = []  # 用来记录所有agent的动作
        # actor_dims = [7, 7, 7, 7, 7, 7, 7, 7]
        for i in range(self.n_agents):
            self.actor_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i]))) # 8 * 50000 * 7
            self.actor_new_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))

    def store_tansition(self, raw_obs, state, action, reward, raw_obs_, state_):
        index = self.mem_cntr % self.mem_size # 丢掉最新进来的机制

        for agent_idx in range(self.n_agents):  # actor 通过 obs得到 actor-state
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            # self.actor_action_memory[agent_idx][index] = action[agent_idx]
        if self.mem_cntr <= self.mem_size:
            self.action_memory.append(action)
        else:
            self.action_memory[index] = action
        self.state_memory[index] = state  # 全局的state就是state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.mem_cntr += 1

    def sample_buffer(self):
        x = random.randrange(0, self.mem_size - self.batch_size)

        states = self.state_memory[x: x + self.batch_size]
        states_ = self.new_state_memory[x: x + self.batch_size]
        rewards = self.reward_memory[x: x + self.batch_size]
        actions = self.action_memory[x: x + self.batch_size]
        actor_states = []
        actor_new_states = []
        for agent_index in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_index][x: x + self.batch_size]) # 从第i个agent的50000个选300个
            actor_new_states.append(self.actor_new_state_memory[agent_index][x: x + self.batch_size])
            # actions.append(self.actor_action_memory[agent_index][x: x + self.batch_size])
        return actor_states, states, actions, rewards, actor_new_states, states_

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
        return False

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name,
                 chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(26 + 32, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state, action):
        state = state.view(-1, 26)
        action = action.view(-1, 32)  # action是onehot？ 对，softmax输出对pi
        x = F.relu(self.fc1(T.cat((state, action), dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

    def init(self):
        self.fc1.weight.data = T.abs(0.01 * T.randn(self.fc1.weight.size()))
        self.fc2.weight.data = T.abs(0.01 * T.randn(self.fc2.weight.size()))
        self.q.weight.data = T.abs(0.01 * T.randn(self.q.weight.size()))

        self.fc1.bias.data = (T.ones(self.fc1.bias.size()) * 0.01)
        self.fc2.bias.data = (T.ones(self.fc2.bias.size()) * 0.01)
        self.q.bias.data = (T.ones(self.q.bias.size()) * 0.01)

    def save_checkpoint(self):
        # print('...saving checkpoint...')
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        # print('...loading checkpoint...')
        self.load_state_dict(T.load(self.chkpt_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)  # 输出的大小是n_actions

        self.optimizerr = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        state = state.view(-1, 26)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.relu(self.pi(x))  # 视频50分50秒对此有解释

        return pi

    def init(self):
        self.fc1.weight.data = T.abs(0.01 * T.randn(self.fc1.weight.size()))
        self.fc2.weight.data = T.abs(0.01 * T.randn(self.fc2.weight.size()))
        self.pi.weight.data = T.abs(0.01 * T.randn(self.pi.weight.size()))

        self.fc1.bias.data = (T.ones(self.fc1.bias.size()) * 0.01)
        self.fc2.bias.data = (T.ones(self.fc2.bias.size()) * 0.01)
        self.pi.bias.data = (T.ones(self.pi.bias.size()) * 0.01)

    def save_checkpoint(self):
        # print('...saving checkpoint...')
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        # print('...loading checkpoint...')
        self.load_state_dict(T.load(self.chkpt_file))


class Agent:  # 代表了每一个agent
    def __init__(self, actor_dims, critic_dims, n_actions, agent_idx, chkpt_dir, n_agents,
                 alpha=0.01, beta=0.01, fc1 = 64, fc2 = 64, gamma=0.8, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx  # 百分号 agent_idx会替换s
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                  chkpt_dir=chkpt_dir, name=self.agent_name + '_actor')
        self.actor.init()
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions,
                                    chkpt_dir=chkpt_dir, name=self.agent_name + '_critic')
        self.critic.init()
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                         chkpt_dir=chkpt_dir, name=self.agent_name + '_target_actor')
        self.target_actor.init()
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions,
                                           chkpt_dir=chkpt_dir, name=self.agent_name + '_target_critic')
        self.target_critic.init()
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)  # type: ignore

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)  # type: ignore

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float)
        action = self.actor.forward(state)
        # noise = T.rand(self.n_actions).to(self.actor.device)
        # action = action + noise

        return action.detach().cpu().numpy()[0]

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, chkpt_dir):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions[agent_idx], agent_idx,
                                     chkpt_dir=chkpt_dir, n_agents=n_agents))

    def initial(self, actor_dims, critic_dims, chkpt_dir):
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions[agent_idx], agent_idx,
                                     chkpt_dir=chkpt_dir, n_agents=n_agents))
        for i in self.agents:
            i.actor.init()
            i.critic.init()
            i.target_actor.init()
            i.target_critic.init()

    def save_checkpoint(self):
        # print('...saving checkpoint...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        # print('...loading checkpoint...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):  # 共有8个obs
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def obs_list_to_state_vector(self, observation):  # 这个是把所有的和在一起
        state = np.array([])
        for obs in observation:
            state = np.concatenate([state, obs])
        return state

def random_pick(some_list, probabilities):
    item = 0
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item

# 之所以observe比随机好，是因为，只要有一点规律，就会还可以，不拉胯。
fc1 = 64
fc2 = 64
alpha = 1e-4
beta = 1e-4
chkpt_dir = '/Users/ocean/git/ECOC-PTP/multi-agent'
n_agents = 2
actor_dims = [26, 26]
critic_dims =  26  # 请求信息 8 * 3 + request 2
n_actions = [16, 16]
INITIAL_EPSILON = 0.6
FINAL_EPSILON = 0.001
epsilon = copy.deepcopy(INITIAL_EPSILON)

# 这个maddpg_agents是关键
maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, chkpt_dir)
memory = MultiAgentReplayBuffer(100, critic_dims, actor_dims, n_actions, n_agents, batch_size=50)

# evaluate = False  # 用于训练完了之后看看模型效果的, 或者继续训练
timer = 0
Observe = 200
Explore = 50000
Train = 30000
# if evaluate:
#     maddpg_agents.load_checkpoint()
requests = [random.randrange(1, 5, 1), random.randrange(1, 5, 1)] # 初始化第一步
slot = {'0': [0, 0, 'M'], '1': [0, 0, 'M'], '2': [0, 0, 'M'], '3': [0, 0, 'M'], '4': [0, 0, 'M'], '5': [0, 0, 'M'], '6': [0, 0, 'M'], '7': [0, 0, 'M']} # 这个也得是drl先给一个初始值
# 初始化第一步
[s, two_actions, r, next_s, requests_next] = action_to_reward([15, 7], slot, requests) # 第一个动作随便选择的
state = s # 预先激活的点，请求的内容
obs = [s, s]  # 每个的状态；state是最后输入评论员的
rewarddd = []  # 用户rewrd的画图

while timer < Observe:
    actions = []  # 这是是两个值，一个是
    action_index = []
    reward = []
    statex = None
    for i in range(2):
        action = list(np.zeros(n_actions[i]))
        rangdom_pick_action = random.randrange(0, n_actions[i], 1)
        action[rangdom_pick_action] = 1
        actions.append(action)
        action_idx = action.index(max(action))
        action_index.append(action_idx)

    [s, three_actions, r, next_s, requests_next] = action_to_reward(action_index, slot, requests_next)
    obs = [s,s]
    state = s
    # print(state)
    state_ = next_s
    obs_ = [next_s, next_s]
    rewardd = r
    memory.store_tansition(obs, state, actions, rewardd, obs_, state_)  # 存的过程，这段时间只会往里存


    if timer <= Observe:
        process = 'observe'
    else:
        process = 'train'
#
#
    if timer % 5 == 0:  # 4500的话是真实的第500个
        sss = 'time_step {}/ process {}/ action {}/ reward {}/ state {}/'.format(timer, process, action_index, rewardd, state)
        print(sss)
        requestANDstate = 'time_step {}/ process {}/ action {}/ reward {}/ state {}/'.format(timer, process, action_index, rewardd, state)
        f = open('/Users/ocean/git/ECOC-PTP/multi-agent/Results/information1', 'a')
        ff = open('/Users/ocean/git/ECOC-PTP/multi-agent/Results/information2','a')
        f.write(sss + '\n')
        ff.write(requestANDstate + '\n')
        f.close()
        ff.close()

    if timer % 500 == 0:
        maddpg_agents.save_checkpoint()

    if timer % 50 == 0:
        rewarddd.append(rewardd)

    if timer % 1000 == 0:
        data_reward = {'reward': rewarddd}
        data_reward = pd.DataFrame(data_reward)
        data_reward.to_csv('/Users/ocean/git/ECOC-PTP/multi-agent/Results/multiagent_data_reward')

    timer += 1
# #
while timer >=  Observe and timer < (Observe + Explore + Train):
    actions_2 = []
    reward = []
    ex_actions = []
    action_index = []
    rdn = random.random()
    learn = 1
    if rdn <= epsilon:
        for i in range(2):
            action_ = list(np.zeros(n_actions[i]))
            rangdom_pick_action = random.randrange(0, n_actions[i], 1)
            action_[rangdom_pick_action] = 1
            actions_2.append(action_)
            action_index.append(rangdom_pick_action)
            learn = 0
    else:
        acts = maddpg_agents.choose_action(obs)  # 所有actor的动作
        acts = list(acts)
        for i in range(2):
            ac = acts[i]
            ac = list(np.array(ac))
            ac_index = np.argmax(ac)
            action_ = list(np.zeros(n_actions[i]))
            action_[ac_index] = 1
            actions_2.append(action_)
            action_index.append(ac_index)

    if epsilon > FINAL_EPSILON and timer > Observe:
        epsilon = epsilon - ((INITIAL_EPSILON - FINAL_EPSILON) / (Explore))

    [s, three_actions, r, next_s, requests_next] = action_to_reward(action_index, slot, requests_next)
    obs = [s, s]
    state = s

    state_ = next_s
    obs_ = [next_s, next_s]
    rewardd = r
    memory.store_tansition(obs, state, actions_2, rewardd, obs_, state_)  # 存的过程，这段时间只会往里存
    timer += 1
    loss = []
    actor_states, states, actions, rewards, actor_new_states, states_ = memory.sample_buffer()

    actions = T.tensor(actions, dtype=T.float)
    states = T.tensor(states, dtype=T.float)

    rewards = T.tensor(rewards, dtype=T.float)
    states_ = T.tensor(states_, dtype=T.float)

    # print(action1)
    all_agents_new_actions = []  # [[batch * 20],[],,,]
    all_agents_new_mu_actions = []

    for agent_idx in range(len(maddpg_agents.agents)):
        new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float)
        new_pi = maddpg_agents.agents[agent_idx].target_actor.forward(new_states)
        all_agents_new_actions.append(new_pi)
        mu_states = T.tensor(actor_states[agent_idx], dtype=T.float)
        pi = maddpg_agents.agents[agent_idx].actor.forward(mu_states)
        all_agents_new_mu_actions.append(pi)

    # print(len(all_agents_new_actions))
    new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)

    mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)

    xxx = 0
    for agent_idx in range(len(maddpg_agents.agents)):
        critic_value_ = maddpg_agents.agents[agent_idx].target_critic.forward(states_, new_actions).flatten()
        critic_value = maddpg_agents.agents[agent_idx].critic.forward(states, actions).flatten()  # 这就是评论员打的分
        target = rewards[:, agent_idx] + maddpg_agents.agents[agent_idx].gamma * critic_value_
        critic_loss = F.mse_loss(target, critic_value)
        maddpg_agents.agents[agent_idx].critic.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        maddpg_agents.agents[agent_idx].critic.optimizer.step()

        actor_loss = maddpg_agents.agents[agent_idx].critic.forward(states, mu).flatten()
        actor_loss = -T.mean(actor_loss)

        maddpg_agents.agents[agent_idx].actor.optimizerr.zero_grad()
        actor_loss.backward(retain_graph=True)

        maddpg_agents.agents[agent_idx].actor.optimizerr.step()
        maddpg_agents.agents[agent_idx].update_network_parameters(tau=0.01)


    if timer <= Observe:
        process = 'observe'
    else:
        process = 'train'
    # print(reward)
    # print(Offload.compresource)
    if loss == []:
        lo = []
    else:
        lo = loss[-1]

    if timer % 5 == 0:  # 4500的话是真实的第500个
        sss = 'time_step {}/ process {}/ action {}/ reward {}/ learn {}/ epsilon {}/ state {}/'.format(timer, process, action_index, rewardd, learn, epsilon, state)
        print(sss)
        requestANDstate = 'time_step {}/ process {}/ action {}/ reward {}/ state {}/'.format(timer, process,action_index, rewardd, state)
        f = open('/Users/ocean/git/ECOC-PTP/multi-agent/Results/information1', 'a')
        ff = open('/Users/ocean/git/ECOC-PTP/multi-agent/Results/information2', 'a')
        f.write(sss + '\n')
        ff.write(requestANDstate + '\n')
        f.close()
        ff.close()

    if timer % 500 == 0:
        maddpg_agents.save_checkpoint()

    if timer % 5 == 0:
        rewarddd.append(rewardd)

    if timer % 500 == 0:
        data_reward = {'reward': rewarddd}
        data_reward = pd.DataFrame(data_reward)
        data_reward.to_csv('/Users/ocean/git/ECOC-PTP/multi-agent/Results/data_reward')


# plot the figure
# check_data_reward = pd.read_csv('/Users/ocean/git/ECOC-PTP/multi-agent/Results/data_reward')
# r = check_data_reward['reward']
# plt.plot(r)



