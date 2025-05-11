import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import matplotlib
import json
from tqdm import tqdm
matplotlib.use('Agg')
plt.rcParams['axes.unicode_minus'] = False  # 绘图显示负号

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--mode", type=str, default="train") # [train, test]
parser.add_argument("--tot_agents_num", type=int, default=5) # Max 5
parser.add_argument("--intelligence_agents_num", type=int, default=1)
args = parser.parse_args()

# 复制环境类
class Env:
    def __init__(self, num_firms, p, h, c, initial_inventory, poisson_lambda=10, max_steps=100):
        """
        初始化供应链管理仿真环境。
        
        :param num_firms: 企业数量
        :param p: 各企业的价格列表
        :param h: 库存持有成本
        :param c: 损失销售成本
        :param initial_inventory: 每个企业的初始库存
        :param poisson_lambda: 最下游企业需求的泊松分布均值
        :param max_steps: 每个episode的最大步数
        """
        self.num_firms = num_firms
        self.p = p  # 企业的价格列表
        self.h = h  # 库存持有成本
        self.c = c  # 损失销售成本
        self.poisson_lambda = poisson_lambda  # 泊松分布的均值
        self.max_steps = max_steps  # 每个episode的最大步数
        self.initial_inventory = initial_inventory  # 初始库存
        
        # 初始化库存
        self.inventory = np.full((num_firms, 1), initial_inventory)
        # 初始化订单量
        self.orders = np.zeros((num_firms, 1))
        # 初始化已满足的需求量
        self.satisfied_demand = np.zeros((num_firms, 1))
        # 记录当前步数
        self.current_step = 0
        # 标记episode是否结束
        self.done = False

    def reset(self):
        """
        重置环境状态。
        """
        self.inventory = np.full((self.num_firms, 1), self.initial_inventory)
        self.orders = np.zeros((self.num_firms, 1))
        self.satisfied_demand = np.zeros((self.num_firms, 1))
        self.current_step = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        """
        获取每个企业的观察信息，包括订单量、满足的需求量和库存。
        每个企业的状态是独立的，包括自己观察的订单、需求和库存。
        """
        return np.concatenate((self.orders, self.satisfied_demand, self.inventory), axis=1)

    def _generate_demand(self):
        """
        根据规则生成每个企业的需求。
        最下游企业的需求遵循泊松分布，其他企业的需求等于下游企业的订单量。
        """
        demand = np.zeros((self.num_firms, 1))
        for i in range(self.num_firms):
            if i == 0:
                # 最下游企业的需求遵循泊松分布，均值为 poisson_lambda
                demand[i] = np.random.poisson(self.poisson_lambda)
            else:
                # 上游企业的需求等于下游企业的订单量
                demand[i] = self.orders[i - 1]  # d_{i+1,t} = q_{it}
        return demand

    def step(self, actions):
        """
        执行一个时间步的仿真，根据给定的行动 (每个企业的订单量) 更新环境状态。
        
        :param actions: 每个企业的订单量 (shape: (num_firms, 1))，即每个智能体的行动
        :return: next_state, reward, done
        """
        self.orders = actions  # 更新订单量
        
        # 生成各企业的需求
        self.demand = self._generate_demand()

        # 计算每个企业收到的订单量和满足的需求
        for i in range(self.num_firms):
            if i == 0:
                # 第一企业从外部需求直接得到满足
                self.satisfied_demand[i] = min(self.demand[i], self.inventory[i])
            else:
                # 后续企业的需求由上游企业订单决定
                self.satisfied_demand[i] = min(self.demand[i], self.inventory[i])
        
        # 更新库存
        for i in range(self.num_firms):
            self.inventory[i] = self.inventory[i] + self.orders[i] - self.satisfied_demand[i]
        
        # 计算每个企业的奖励: p_i * d_{it} - p_{i+1} * q_{it} - h * I_{it}
        rewards = np.zeros((self.num_firms, 1))
        loss_sales = np.zeros((self.num_firms, 1))  # 损失销售费用
        
        for i in range(self.num_firms):
            rewards[i] += self.p[i] * self.satisfied_demand[i] - (self.p[i+1] if i+1 < self.num_firms else 0) * self.orders[i] - self.h * self.inventory[i]
            
            # 损失销售计算
            if self.satisfied_demand[i] < self.demand[i]:
                loss_sales[i] = (self.demand[i] - self.satisfied_demand[i]) * self.c
        
        rewards -= loss_sales  # 总奖励扣除损失销售成本
        
        # 增加步数
        self.current_step += 1
        
        # 判断是否结束（比如达到最大步数）
        if self.current_step >= self.max_steps:
            self.done = True
        
        return self._get_observation(), rewards, self.done

# 定义Q网络模型
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        初始化Q网络
        
        :param state_size: 状态空间维度
        :param action_size: 动作空间维度
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """
        前向传播
        
        :param state: 输入状态
        :return: 各动作的Q值
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        """
        初始化经验回放缓冲区
        
        :param capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        添加经验到缓冲区
        
        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param next_state: 下一个状态
        :param done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        从缓冲区采样一批经验
        
        :param batch_size: 批大小
        :return: 一批经验 (state, action, reward, next_state, done)
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """
        获取缓冲区当前大小
        
        :return: 缓冲区大小
        """
        return len(self.buffer)

# 定义DQN智能体
class DQNAgent:
    def __init__(self, state_size, action_size, firm_id, max_order=20, buffer_size=10000, batch_size=64, gamma=0.99, 
                 learning_rate=1e-3, tau=1e-3, update_every=4, device="cpu"):
        """
        初始化DQN智能体
        
        :param state_size: 状态空间维度
        :param action_size: 动作空间维度
        :param firm_id: 企业ID，用于标识训练哪个企业
        :param max_order: 最大订单量，用于离散化动作空间
        :param buffer_size: 回放缓冲区大小
        :param batch_size: 批大小
        :param gamma: 折扣因子
        :param learning_rate: 学习率
        :param tau: 软更新参数
        :param update_every: 更新目标网络的频率
        """
        self.state_size = state_size
        self.action_size = action_size
        self.firm_id = firm_id
        self.max_order = max_order
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.learning_step = 0
        self.device = device
        
        # 创建Q网络和目标网络
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 设置优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 创建经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size)
        
        # 跟踪训练进度
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        """
        添加经验到回放缓冲区并按需学习
        
        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param next_state: 下一个状态
        :param done: 是否结束
        """
        # 添加经验到回放缓冲区
        self.memory.add(state, action, reward, next_state, done)
        
        # 每隔一定步数学习
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
    
    def act(self, state, epsilon=0.0):
        """
        根据当前状态选择动作
        
        :param state: 当前状态
        :param epsilon: epsilon-贪婪策略参数
        :return: 选择的动作
        """
        # 从3维numpy数组转换为1维向量
        state = torch.from_numpy(state.flatten()).float().unsqueeze(0).to(self.device)
        
        # 切换到评估模式
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        # 切换回训练模式
        self.q_network.train()
        
        # epsilon-贪婪策略
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy()) + 1  # +1 因为我们的动作从1开始
        else:
            return random.randint(1, self.max_order)
    
    def learn(self, experiences):
        """
        从经验批次中学习
        
        :param experiences: (state, action, reward, next_state, done) 元组
        """
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # 转换为torch张量
        states = torch.from_numpy(np.vstack([s.flatten() for s in states])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([a-1 for a in actions])).long().to(self.device)  # -1 因为我们的动作从1开始，但索引从0开始
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([ns.flatten() for ns in next_states])).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
        
        # 从目标网络获取下一个状态的最大预测Q值
        Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        
        # 计算目标Q值
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # 获取当前Q值估计
        Q_expected = self.q_network(states).gather(1, actions)
        
        # 计算损失
        loss = nn.MSELoss()(Q_expected, Q_targets)
        
        # 最小化损失
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.learning_step += 1
        if self.learning_step % self.update_every == 0:
            self.soft_update()
        
        return loss.item()
    
    def soft_update(self):
        """
        软更新目标网络参数：θ_target = τ*θ_local + (1-τ)*θ_target
        """
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filename):
        """
        保存模型参数
        
        :param filename: 文件名
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 保存模型状态字典
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        print(f"模型已保存到 {filename}")
    
    def load(self, filename):
        """
        加载模型参数
        
        :param filename: 文件名
        """
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"从 {filename} 加载了模型")
            return True
        return False

def train_dqn(env: Env, agents, num_episodes=1000, max_t=100, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    训练DQN智能体
    
    :param env: 环境
    :param agents: DQN多智能体
    :param num_episodes: 训练的episodes数量
    :param max_t: 每个episode的最大步数
    :param eps_start: 起始epsilon值
    :param eps_end: 最小epsilon值
    :param eps_decay: epsilon衰减率
    :return: 所有episode的奖励
    """
    scores = {i: [] for i in agents}  # 每个episode的总奖励
    eps = eps_start  # 初始epsilon值
    
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        score = {i: 0 for i in agents}
        
        for t in range(max_t):
            # 对特定企业采取动作，其他企业随机决策
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                if firm_id in agents:
                    # 使用智能体策略
                    firm_state = state[firm_id].reshape(1, -1)
                    action = agents[firm_id].act(firm_state, eps)
                    actions[firm_id] = action
                else:
                    # 对其他企业采取随机策略
                    actions[firm_id] = np.random.randint(1, 21)
            
            # 执行动作
            next_state, rewards, done = env.step(actions)
            
            for firm_id in range(env.num_firms):
                if firm_id in agents:
                    # 该企业的奖励
                    reward = rewards[firm_id][0]
            
                    # 保存经验并学习
                    agents[firm_id].step(state[firm_id].reshape(1, -1), actions[firm_id], reward, next_state[firm_id].reshape(1, -1), done)
                    
                    # 更新状态和奖励
                    state = next_state
                    score[firm_id] += reward
            
            if done:
                break
        
        # 更新epsilon
        eps = max(eps_end, eps_decay * eps)
        
        # 记录分数
        for i in agents:
            scores[i].append(score[i])
        
        # 输出进度
        if i_episode % 100 == 0:
            for i in agents:
                print(f'Episode {i_episode}/{num_episodes} | Agent {i} | Average Score: {np.mean(scores[i][-100:]):.2f} | Epsilon: {eps:.4f}')
        
        # 每隔一定episode保存模型
        if i_episode % 500 == 0:
            for i in agents:
                agents[i].save(f'models/marl_{args.tot_agents_num}_{args.intelligence_agents_num}/agent_firm_{i}_episode_{i_episode}.pth')
    
    # 训练结束后保存最终模型
    for i in agents:
        agents[i].save(f'models/marl_{args.tot_agents_num}_{args.intelligence_agents_num}/agent_firm_{i}_final.pth')
    
    return scores

def test_agent(env, agents, num_episodes=10):
    """
    测试训练好的DQN智能体
    
    :param env: 环境
    :param agent: 训练好的DQN智能体
    :param num_episodes: 测试的episodes数量
    :return: 所有episode的奖励和详细信息
    """
    scores = {i: [] for i in agents}
    inventory_history = {i: [] for i in agents}
    orders_history = {i: [] for i in agents}
    demand_history = {i: [] for i in agents}
    satisfied_demand_history = {i: [] for i in agents}
    demand_complete_history = {i: [] for i in agents}
    
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        score = {i: 0 for i in agents}
        episode_inventory = {i: [] for i in agents}
        episode_orders = {i: [] for i in agents}
        episode_demand = {i: [] for i in agents}
        episode_satisfied_demand = {i: [] for i in agents}
        episode_demand_complete = {i: [] for i in agents}
        
        for t in range(env.max_steps):
            # 对特定企业采取动作，其他企业随机决策
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                if firm_id in agents:
                    # 使用智能体策略，不使用探索
                    firm_state = state[firm_id].reshape(1, -1)
                    action = agents[firm_id].act(firm_state, epsilon=0.0)
                    actions[firm_id] = action
                else:
                    # 对其他企业采取随机策略
                    actions[firm_id] = np.random.randint(1, 21)
            
            # 执行动作
            next_state, rewards, done = env.step(actions)
            
            # 记录关键指标
            for i in agents:
                episode_inventory[i].append(env.inventory[i][0])
                episode_orders[i].append(actions[i][0])
                episode_demand[i].append(env.demand[i][0])
                episode_satisfied_demand[i].append(env.satisfied_demand[i][0])
                episode_demand_complete[i].append(env.satisfied_demand[i][0] / env.demand[i][0])
            
                # 该企业的奖励
                reward = rewards[i][0]
                score[i] += reward
            
            # 更新状态
            state = next_state
            
            if done:
                break
        
        # 记录分数和历史数据
        for i in agents:
            scores[i].append(score[i])
            inventory_history[i].append(episode_inventory[i])
            orders_history[i].append(episode_orders[i])
            demand_history[i].append(episode_demand[i])
            satisfied_demand_history[i].append(episode_satisfied_demand[i])
            demand_complete_history[i].append(episode_demand_complete[i])
        
            print(f'Test Episode {i_episode}/{num_episodes} | Agent {i} | Score: {score[i]:.2f}')
    
    return scores, inventory_history, orders_history, demand_history, satisfied_demand_history, demand_complete_history

def plot_training_results(scores, window_size=100):
    """
    绘制训练结果
    
    :param scores: 每个episode的奖励
    :param window_size: 移动平均窗口大小
    """
    # 计算移动平均
    def moving_average(data, window_size):
        return [np.mean(data[max(0, i-window_size):i+1]) for i in range(len(data))]
    
    plt.figure(figsize=(10, 6))
    plt.title('Rewards During DQN Training')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    for i in scores:
        avg_scores = moving_average(scores[i], window_size)
        plt.plot(np.arange(len(scores[i])), scores[i], alpha=0.1, label=f'Raw Reward {i}')
        plt.plot(np.arange(len(avg_scores)), avg_scores, label=f'{window_size}-episode Moving Average {i}')
        
    plt.legend()
    plt.savefig(f'figures/marl_{args.tot_agents_num}_{args.intelligence_agents_num}/agent_firm_training_rewards.svg')
    plt.close()

def plot_test_results(scores, inventory_history, orders_history, demand_history, satisfied_demand_history, demand_complete_history):
    """
    绘制测试结果
    
    :param scores: 每个episode的奖励
    :param inventory_history: 每个episode的库存历史
    :param orders_history: 每个episode的订单历史
    :param demand_history: 每个episode的需求历史
    :param satisfied_demand_history: 每个episode的满足需求历史
    """
    tot_results = {}
    for i in scores:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        # 计算平均值，用于绘图
        avg_inventory = np.mean(inventory_history[i], axis=0)
        avg_orders = np.mean(orders_history[i], axis=0)
        avg_demand = np.mean(demand_history[i], axis=0)
        avg_satisfied_demand = np.mean(satisfied_demand_history[i], axis=0)
        avg_demand_complete = np.mean(demand_complete_history[i], axis=0)

        tot_results[i] = {
            "avg_demand_complete": np.mean(avg_demand_complete),
            "avg_orders_fluctuation": np.mean([abs(a-b) for a, b in zip(avg_orders[1:], avg_orders[:-1])])
        }
        
        # 创建图表
        # Orders plot
        axs[0, 0].plot(avg_orders)
        axs[0, 0].set_title('Average Order Quantity')
        axs[0, 0].set_xlabel('Time Step')
        axs[0, 0].set_ylabel('Order Quantity')

        # Demand Complete plot
        axs[0, 1].plot(avg_demand_complete, label=f'Firm {i}')
        axs[0, 1].set_title('Average Demand Complete Rate')
        axs[0, 1].set_xlabel('Time Step')
        axs[0, 1].set_ylabel('Demand Complete Rate')

        # Demand vs. Satisfied Demand plot
        axs[1, 0].plot(avg_demand, label='Demand')
        axs[1, 0].plot(avg_satisfied_demand, label='Satisfied Demand')
        axs[1, 0].set_title('Average Demand vs. Satisfied Demand')
        axs[1, 0].set_xlabel('Time Step')
        axs[1, 0].set_ylabel('Quantity')
        axs[1, 0].legend()

        # Reward bar chart
        axs[1, 1].bar(range(len(scores[i])), scores[i])
        axs[1, 1].set_title('Test Episode Rewards')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Total Reward')
    
        plt.tight_layout()
        plt.savefig(f'figures/marl_{args.tot_agents_num}_{args.intelligence_agents_num}/agent_firm_{i}_test_results.svg')
        plt.close()
    
    with open(f"figures/marl_{args.tot_agents_num}_{args.intelligence_agents_num}/test_results.json", "w") as f:
        json.dump(tot_results, f, indent=4)
    return 

if __name__ == "__main__":
    # 创建保存模型和图表的目录
    os.makedirs(f'models/marl_{args.tot_agents_num}_{args.intelligence_agents_num}', exist_ok=True)
    os.makedirs(f'figures/marl_{args.tot_agents_num}_{args.intelligence_agents_num}', exist_ok=True)
    
    # 初始化环境参数
    num_firms = args.tot_agents_num  # 假设有3个企业
    p = [20 - 4 * i for i in range(num_firms)]  # 价格列表
    h = 0.5  # 库存持有成本
    c = 2  # 损失销售成本
    initial_inventory = 20  # 初始库存
    poisson_lambda = 10  # 泊松分布的均值
    max_steps = 100  # 每个episode的最大步数
    state_size = 3  # 每个企业的状态维度：订单、满足的需求和库存
    action_size = 20  # 假设最大订单量为20
    num_episodes = 5000
    test_episodes = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建仿真环境
    env = Env(num_firms, p, h, c, initial_inventory, poisson_lambda, max_steps)
    agents = {}
    
    if args.mode == "train":
        for firm_id in range(args.tot_agents_num-1, args.tot_agents_num-args.intelligence_agents_num-1, -1):
            agent = DQNAgent(state_size=state_size, action_size=action_size, firm_id=firm_id, max_order=action_size, device=device)
            agents[firm_id] = agent
        
        # 训练DQN智能体
        scores = train_dqn(env, agents, num_episodes=num_episodes, max_t=max_steps, eps_start=1.0, eps_end=0.01, eps_decay=0.995)
        plot_training_results(scores)
    else:
        for firm_id in range(args.tot_agents_num-1, args.tot_agents_num-args.intelligence_agents_num-1, -1):
            agent = DQNAgent(state_size=state_size, action_size=action_size, firm_id=firm_id, max_order=action_size, device=device)
            agent.load(f'models/marl_{args.tot_agents_num}_{args.intelligence_agents_num}/agent_firm_{firm_id}_final.pth')
            agents[firm_id] = agent

    # 测试训练好的智能体
    test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history, demand_complete_history = test_agent(env, agents, num_episodes=test_episodes)
    
    # 绘制测试结果
    plot_test_results(test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history, demand_complete_history)
