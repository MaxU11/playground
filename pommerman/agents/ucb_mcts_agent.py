# The UCB MCTS agent

import math
import random

from collections import defaultdict
from .abstract_mcts_agent import AbstractMCTSAgent
from .env_simulator import EnvSimulator


class UcbMCTSAgent(AbstractMCTSAgent):
    """The UCB-MCTS Agent."""

    def __init__(self, *args, **kwargs):
        super(UcbMCTSAgent, self).__init__(*args, **kwargs)
        # parent hyperparameter
        self.expandTreeRollout = kwargs.get('expandTreeRollout', False)
        self.maxIterations = kwargs.get('maxIterations', 1000)
        self.maxTime = kwargs.get('maxTime', 0.1)
        # hyperparameter
        self.discountFactor = kwargs.get('discountFactor', 0.9999)
        self.depthLimit = kwargs.get('depthLimit', 26)
        self.C = kwargs.get('C', 0.5) # exploration weight

        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node

    def agent_reset(self):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        AbstractMCTSAgent.agent_reset(self)

    def root_changed(self, root):
        # signal that root has changed
        if root is None:
            self.Q = defaultdict(int)
            self.N = defaultdict(int)

    def get_selected_child(self, node):
        # select child for traversing using UCB
        log_n = self.N[node]

        def ucb(a):
            child = node.children[a]
            v_n = self.Q[child] / self.N[child]  # value of node
            if node.agent_id is self.agent_id:
                return v_n + self.C * math.sqrt(log_n / self.N[child])
            else:
                return v_n - self.C * math.sqrt(log_n / self.N[child])

        if node.agent_id is self.agent_id:
            action = max(node.children, key=ucb)
        else:
            action = min(node.children, key=ucb)
        return node.children[action]

    def get_my_expand_action(self, node):
        # return action from my agent
        return random.choice(node.unseen_actions)

    def get_enemy_expand_action(self, node):
        # return action from my agent
        return random.choice(node.unseen_actions)

    def get_my_rollout_action(self, node, data):
        # return action from my agent
        #if node.agent_id == 0:
        #    data.simulation_bomb_life = 2
        return random.choice(node.unseen_actions)

    def get_enemy_rollout_action(self, node, data):
        # return action from my agent
        return random.choice(node.unseen_actions)

    def result(self, node, data):
        # get reward from terminal node
        reward = 0.0
        alive = EnvSimulator.get_alive(data)
        for a in alive:
            if a == self.agent_id:
                if alive[a]:
                    reward += 1.0
                else:
                    reward += -1.0
            else:
                if alive[a]:
                    reward += -0.5
                else:
                    reward += 0.5
        return reward

    def update_stats(self, node, action, result):
        # get updated node stats
        self.N[node] += 1
        self.Q[node] += result

        node.reward = self.Q[node] / self.N[node]
        return result * self.discountFactor

    def best_child(self, node):
        # pick child with highest number of visits
        def score(a):
            child = node.children[a]
            if self.N[child] == 0:
                print('pick of unvisited childs!!')
                return 0 # float("-inf")
            score = self.Q[child] / self.N[child]
            return score

        if len(node.children) == 0:
            print("No children available!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!?????", self.root)
            return 0

        if node.agent_id == self.agent_id:
            return max(node.children, key=score)
        else:
            return min(node.children, key=score)

    def get_agent_map_info(self, node):
        info = 'reward: {:.2f} N:{} Q:{:.2f} (visits '.format(node.reward, self.N[node], self.Q[node])
        for a in node.children:
            info += '{}: {}|{:.2f}, '.format(a, self.N[node.children[a]], self.Q[node.children[a]]/self.N[node.children[a]])
        info += ')'
        return info

    def non_terminal(self, node):
        # check if node is terminal
        if self.depthLimit and (node.depth - self.root.depth) >= self.depthLimit:
            return False

        return AbstractMCTSAgent.non_terminal(self, node)
