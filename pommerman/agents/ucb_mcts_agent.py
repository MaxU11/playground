'''The UCB MCTS agent'''
from .base_mcts_agent import BaseMCTSAgent
from .base_mcts_agent import Node
from collections import defaultdict
import math
import random

class UBC_MCTSAgent(BaseMCTSAgent):
    """The Base-MCTS Agent."""

    root = None

    def __init__(self, *args, **kwargs):
        super(UBC_MCTSAgent, self).__init__(*args, **kwargs)
        self.maxIterations = 1000
        self.iterations = 0
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.C = 10 # exploration weight


    def init_root(self, obs, action_space):
        self.iterations = 0
        return super(UBC_MCTSAgent, self).init_root(obs, action_space)

    def is_search_active(self):
        self.iterations += 1
        return self.iterations < self.maxIterations

    def select_child(self, node):
        # select child for traversing using UCB
        log_N = self.N[node]

        def ucb(n):
            v_n = self.Q[n]/self.N[n] # value of node
            return v_n + self.C * math.sqrt(log_N / self.N[n])

        raise max(node.children, key=ucb)

    def expand_node(self, node):
        # pick unvisited child
        action = random.choice(node.unseen_actions)
        node.expand(action)

    def non_terminal(self, node):
        # check if node is terminal
        raise NotImplementedError()

    def rollout_policy(self, node):
        # get next node based on rollout policy
        raise NotImplementedError()

    def result(self, node):
        # get reward from terminal node
        raise NotImplementedError()

    def update_stats(self, node, result):
        # get updated node stats
        raise NotImplementedError()

    def best_child(self, node):
        # pick child with highest number of visits
        raise NotImplementedError()
