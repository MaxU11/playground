'''The MCTS skeleton'''

import datetime

from abc import ABC, abstractmethod
from . import BaseAgent
from .. import characters


class AbstractMCTSSkeleton(ABC, BaseAgent):
    """The MCTS Skeleton."""

    def __init__(self, *args, **kwargs):
        super(AbstractMCTSSkeleton, self).__init__(character=kwargs.get('character', characters.Bomber))
        self._actNum = 0
        self._avgTime = 0
        self._rolloutNum = 0
        self._avgRolloutDepth = 0
        self.action_space = 6

    def reset(self):
        self._actNum = 0
        self._avgTime = 0
        self._rolloutNum = 0
        self._avgRolloutDepth = 0
        self.agent_reset()
        self._character.reset()

    def act(self, obs, action_space):
        self.action_space = action_space

        start_t = datetime.datetime.now()
        iterations = 0
        self._actNum += 1

        root = self.get_root(obs, action_space)
        while self.is_search_active():
            iterations += 1
            leaf = self.traverse(root)
            simulation_result, leaf, leaf_a = self.rollout(leaf)
            self.backpropagate(leaf, leaf_a, simulation_result)

        #self.search_finished()

        if len(root.children) == 0:
            print('no children!!!!', iterations, root.done, root.depth)

        a = self.best_child(root)
        time_diff = datetime.datetime.now() - start_t
        self._avgTime = self._avgTime + (time_diff.total_seconds() - self._avgTime) / self._actNum
        # print(f'player{self.agent_id}: selected action: {a}, time: {time_diff}, iterations: {iterations}, board: \n{obs["board"]}')
        return a

    # function for node traversal
    def traverse(self, node):
        while node.fully_expanded():
            node = self.get_selected_child(node)

        if self.non_terminal_traverse(node):
            return self.expand_node(node)
        else:
            return node

    # function for the result of the simulation
    def rollout(self, node):
        leaf = node
        leaf_action = None
        data = self.get_data(node)
        self._rolloutNum += 1
        depth = 0
        while self.non_terminal_rollout(node):
            depth += 1
            node, data, a = self.rollout_policy(node, data)
            if not leaf_action:
                leaf_action = a

        self._avgRolloutDepth = self._avgRolloutDepth + (depth - self._avgRolloutDepth) / self._rolloutNum
        if len(leaf.children) > 0:
            leaf = node
            leaf_action = None
        return self.result(node, data), leaf, leaf_action

    # function for backpropagation
    def backpropagate(self, node, action, result):
        result = self.update_stats(node, action, result)
        if node.is_root():
            return
        self.backpropagate(node.parent, node.action, result)

    def get_agent_info(self, info):
        info['avgTime'] = self._avgTime
        info['avgRolloutDepth'] = self._avgRolloutDepth

    def non_terminal_traverse(self, node):
        return self.non_terminal(node)

    def non_terminal_rollout(self, node):
        return self.non_terminal(node)

    @abstractmethod
    def agent_reset(self):
        # reset agent
        raise NotImplementedError()

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
    def update_stats(self, node, action, result):
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