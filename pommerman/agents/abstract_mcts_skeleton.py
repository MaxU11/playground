'''The MCTS skeleton'''

import datetime

from abc import ABC, abstractmethod
from . import BaseAgent


class AbstractMCTSSkeleton(ABC, BaseAgent):
    """The MCTS Skeleton."""

    def __init__(self, *args, **kwargs):
        super(AbstractMCTSSkeleton, self).__init__(*args, **kwargs)
        self.expand_tree_rollout = False

    def act(self, obs, action_space):
        start_t = datetime.datetime.now()

        root = self.get_root(obs, action_space)
        while self.is_search_active():
            leaf = self.traverse(root)
            simulation_result, leaf = self.rollout(leaf)
            self.backpropagate(leaf, simulation_result)

        #self.search_finished()

        a = self.best_child(root)
        end_t = datetime.datetime.now()
        print(f'selected action: {a}, time: {end_t - start_t}')
        return a

    # function for node traversal
    def traverse(self, node):
        while node.fully_expanded():
            node = self.get_selected_child(node)

        if self.non_terminal(node):
            return self.expand_node(node)
        else:
            return node

    # function for the result of the simulation
    def rollout(self, node):
        leaf = node
        data = self.get_data(node)
        while self.non_terminal(node):
            node, data = self.rollout_policy(node, data)
        if len(leaf.children) > 0:
            leaf = node
        return self.result(node, data), leaf

    # function for backpropagation
    def backpropagate(self, node, result):
        result = self.update_stats(node, result)
        if node.is_root():
            return
        self.backpropagate(node.parent, result)

    @abstractmethod
    def get_root(self, obs, action_space):
        # initialize root node
        raise NotImplementedError()

    @abstractmethod
    def get_data(self, node):
        # initialize root node
        raise NotImplementedError()

    @abstractmethod
    def is_search_active(self):
        # is search active?
        raise NotImplementedError()

    @abstractmethod
    def get_selected_child(self, node):
        # select child for traversing
        raise NotImplementedError()

    @abstractmethod
    def expand_node(self, node):
        # pick unvisited child
        raise NotImplementedError()

    @abstractmethod
    def non_terminal(self, node):
        # check if node is terminal
        raise NotImplementedError()

    @abstractmethod
    def rollout_policy(self, node, data):
        # get next node based on rollout policy
        raise NotImplementedError()

    @abstractmethod
    def result(self, node, data):
        # get reward from terminal node
        raise NotImplementedError()

    @abstractmethod
    def update_stats(self, node, result):
        # get updated node stats
        raise NotImplementedError()

    @abstractmethod
    def best_child(self, node):
        # pick child with highest number of visits
        raise NotImplementedError()

    @abstractmethod
    def search_finished(self):
        # called when search has finished
        pass

    @abstractmethod
    def create_node(self, depth, game_data, action_space, agent_id, enemy_id, action, save_data=True):
        # create new tree node
        raise NotImplementedError()