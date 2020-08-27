'''The UCB MCTS agent'''
import math
import random

from copy import deepcopy
from collections import defaultdict
from .base_mcts_agent import BaseMCTSAgent
from .base_mcts_agent import Node
from .env_simulator import Env_simulator
from .. import constants

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
        self.discount_factor = 0.999

    def get_root(self, obs, action_space):
        self.iterations = 0
        if (self.root == None):
            game_state = Env_simulator.get_initial_game_state(obs, 10 + self.agent_id)
            self.root = Node(game_state, action_space, 10 + self.agent_id, self.enemies[0].value, None)
        else:
            actions = Env_simulator.update(self.root.game_state, obs)
            child = self.root.get_child(actions)
            if child != None and Env_simulator.boards_equal(self.root.game_state, child.game_state, False):
                self.root = child
            else:
                self.root = Node(self.root.game_state, action_space, 10 + self.agent_id, self.enemies[0].value, None)

        return self.root

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
        if node.agent_id == 10 + self.agent_id:
            action = random.choice(node.unseen_actions)
            node = node.expand(action, node.game_state)

        game_state = deepcopy(node.game_state)
        action = random.choice(node.unseen_actions)
        Env_simulator.act(game_state, {node.enemy_id: node.action, node.agent_id: action})
        node.expand(action, game_state)

    def non_terminal(self, node):
        # check if node is terminal
        return not node.game_state.done

    def rollout_policy(self, node):
        # get next node based on rollout policy
        actions = {}
        if node.agent_id == 10 + self.agent_id:
            actions[10 + self.agent_id] = node.action
        else:
            actions[10 + self.agent_id] = random.choice(range(node.action_space.n))
        actions[self.enemies[0].value] = random.choice(range(node.action_space.n))

        game_state = deepcopy(node.game_state)
        Env_simulator.act(game_state, actions)
        return Node(game_state, node.action_space, 10 + self.agent_id, self.enemies[0], None)

    def result(self, node):
        # get reward from terminal node
        reward = 0
        for a in node.game_state.agents:
            if a.agent_id == 10 + self.agent_id:
                if a.is_alive: reward += 500
                else: reward += 0
            else:
                if a.is_alive: reward += 0
                else: reward += 500
        return reward

    def update_stats(self, node, result):
        # get updated node stats
        self.N[node] += 1
        if node.agent_id == 10 + self.agent_id:
            self.Q[node] += result
        else:
            self.Q[node] += 1000 - result
        result *= self.discount_factor

    def best_child(self, node):
        # pick child with highest number of visits
        def score(child):
            if self.N[child] == 0:
                return float("-inf")
            return self.Q[child] / self.N[child]

        return max(node.children, key=score)
