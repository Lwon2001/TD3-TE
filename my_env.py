import networkx as nx
import numpy as np
import torch

class Env:
    def __init__(self, alpha):
        filename = './data/abilene_edgelist_noweight.txt'  # 使用权重设置为0的边进行创建
        self.G = nx.read_edgelist(filename, create_using=nx.Graph(), nodetype=int, data=(("weight", float),))
        self.node_num = len(self.G.nodes)
        self.link_num = len(self.G.edges.items())
        self.state = torch.zeros([self.node_num,  self.node_num])  # 状态表示为TM需求
        self.node_degree = [len(self.G.degree._nodes[i]) for i in range(1, self.node_num + 1)]
        self.link_capacity = 100
        self.links = [l[0] for l in self.G.edges.items()]
        self.link_util = torch.zeros([self.link_num])
        self.alpha = alpha  # 奖励函数的乘积因子
        self.action_dim = len(self.G.edges)
        self.state_dim = self.node_num * self.node_num
        self.action_low_space = 0.01
        self.action_high_space = 2
        self.best_MLU = 1

    def reset(self):
        self.state = torch.zeros([self.node_num,  self.node_num])
        self.link_util = torch.zeros([self.link_num])
        return self.state.flatten()

    def set_weight(self, action):
        """
        根据agent的action设置网络的链路权重
        :param action:
        :return:
        """
        for i in range(0, self.action_dim):
            node1 = self.links[i][0]
            node2 = self.links[i][1]
            self.G.adj._atlas[node1][node2]['weight'] = action[i]  # 为结点及其邻居之间的边赋新的权重


    def compute_reward(self):
        """
        根据上一状态的TM，采取action的权重设置后，进行ospf路由，得到最大链路利用率，根据该链路利用率来给与相应奖励
        注意，在执行该函数前已经设置action对应的链路权重
        """
        self.link_util = torch.zeros([self.link_num])
        # 遍历所有TM，每个TM按照最短路径路由
        for node1 in range(1, self.node_num + 1):
            for node2 in range(1, self.node_num + 1):
                if node1 != node2:
                    # 为最短路径上的每条链接添加该流量值
                    shortest_path = nx.dijkstra_path(self.G, node1, node2)
                    ptr = 1
                    traffic_demand = self.state[node1-1][node2-1]
                    while ptr < len(shortest_path):
                        # shortest_path[ptr-1] 与 shortest_path[ptr]分别为一条边的两个端节点
                        try:
                            index = self.links.index((shortest_path[ptr-1], shortest_path[ptr]))  # index为对应的边在links中的下标
                        except ValueError:
                            index = self.links.index((shortest_path[ptr], shortest_path[ptr - 1]))
                        self.link_util[index] += traffic_demand  # 在保存对应链路利用率的表中添加数据
                        ptr += 1
        MLU = torch.max(self.link_util / self.link_capacity)
        return -self.alpha * MLU

    def generate_TM(self):
        """根据新的权重生成新的TM，这个过程如果太慢可以尝试用矩阵运算来修改"""
        for i in range(1, self.node_num + 1):
            for j in range(1, self.node_num + 1):
                if i != j:
                    i_degree = self.node_degree[i - 1]
                    j_degree = self.node_degree[j - 1]
                    self.state[i-1][j-1] = i_degree * j_degree / nx.dijkstra_path_length(self.G, i, j)

    def next_state(self, action):
        """
        :param action: agent做出的动作决策，具体表现为链路的权重设置
        :return: next_state,reward,done
        当网络接受到新的权重设置后，根据网络的最大链路利用率计算出reward
        然后根据新的权重设置生成新的TM，即为网络的下一个状态
        done始终为0
        """
        self.set_weight(action)
        reward = self.compute_reward()
        self.generate_TM()
        return [self.state.flatten(), reward, 0]

