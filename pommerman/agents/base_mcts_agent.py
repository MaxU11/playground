'''The base MCTS agent'''
from abc import abstractmethod
from . import BaseAgent

class BaseMCTSAgent(BaseAgent):
    """The Base-MCTS Agent."""

    def __init__(self, *args, **kwargs):
        super(BaseMCTSAgent, self).__init__(*args, **kwargs)
        # iterations the agent is allowed to make for each action
        self.maxIterations = 1000

    def act(self, obs, action_space):
        for i in range(self.maxIterations):
            leaf = self.traverse(obs)
            simulation_result = self.rollout(leaf)
            self.backpropagate(leaf, simulation_result)

        return self.best_child(obs)

    # function for node traversal
    def traverse(self, node):
        while self.fully_expanded(node):
            node = self.select_child(node)

        # in case no children are present / node is terminal
        return self.expand_node(node) or node

    # function for the result of the simulation
    def rollout(self, node):
        while self.non_terminal(node):
            node = self.rollout_policy(node)
        return self.result(node)

    # function for backpropagation
    def backpropagate(self, node, result):
        if self.is_root(node):
            return
        node.stats = self.update_stats(node, result)
        self.backpropagate(node.parent)

    @abstractmethod
    def fully_expanded(self, node):
        # check if fully expanded
        raise NotImplementedError()

    @abstractmethod
    def select_child(self, node):
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
    def rollout_policy(self, node):
        # get next node based on rollout policy
        raise NotImplementedError()

    @abstractmethod
    def result(self, node):
        # get reward from terminal node
        raise NotImplementedError()

    @abstractmethod
    def is_root(self, node):
        # check if node is root
        raise NotImplementedError()

    @abstractmethod
    def update_stats(self, node, result):
        # get updated node stats
        raise NotImplementedError()

    @abstractmethod
    def best_child(self, node):
        # pick child with highest number of visits
        raise NotImplementedError()
