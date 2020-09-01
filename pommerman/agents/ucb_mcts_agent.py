# The UCB MCTS agent
import math
import random

import copy
from collections import defaultdict
from .base_mcts_agent import BaseMCTSAgent
from .base_mcts_agent import Node
from .env_simulator import Env_simulator
from .env_simulator import Game_state
from .. import constants


class UBC_MCTSAgent(BaseMCTSAgent):
    """The Base-MCTS Agent."""

    root = None

    def __init__(self, *args, **kwargs):
        super(UBC_MCTSAgent, self).__init__(*args, **kwargs)
        self.maxIterations = 100
        self.iterations = 0
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.C = 10  # exploration weight
        self.discount_factor = 0.9999

    def root_changed(self, root):
        # signal that root has changed
        self.iterations = 0
        if root is None:
            self.Q = defaultdict(int)
            self.N = defaultdict(int)

    def is_search_active(self):
        self.iterations += 1
        return self.iterations < self.maxIterations

    def get_selected_child(self, node):
        # select child for traversing using UCB
        log_n = self.N[node]

        def ucb(a):
            child = node.children[a]
            v_n = self.Q[child] / self.N[child]  # value of node
            return v_n + self.C * math.sqrt(log_n / self.N[child])

        action = max(node.children, key=ucb)
        return node.children[action]

    def get_my_expand_action(self, node):
        # return action from my agent
        return random.choice(node.unseen_actions)

    def get_enemy_expand_action(self, node):
        # return action from my agent
        return random.choice(node.unseen_actions)

    def get_my_rollout_action(self, node):
        # return action from my agent
        return random.choice(range(node.action_space.n))

    def get_enemy_rollout_action(self, node):
        # return action from my agent
        return random.choice(range(node.action_space.n))

    def result(self, node):
        # get reward from terminal node
        reward = 0.0
        for a in node.game_state.agents:
            if a.agent_id == self.agent_id:
                if a.is_alive:
                    reward += 1000.0
                else:
                    reward += -1000.0
            else:
                if a.is_alive:
                    reward += -500.0
                else:
                    reward += 500.0
        return reward

    def update_stats(self, node, result):
        # get updated node stats
        self.N[node] += 1
        if node.agent_id == self.agent_id:
            self.Q[node] += -result
        else:
            self.Q[node] += result
        node.reward = self.Q[node] / self.N[node]
        return result * self.discount_factor

    def best_child(self, node):
        # pick child with highest number of visits
        def score(a):
            child = node.children[a]
            if self.N[child] == 0:
                return float("-inf")
            return self.Q[child] / self.N[child]

        return max(node.children, key=score)
