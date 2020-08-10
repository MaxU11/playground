'''The base MCTS agent'''
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

        return self.best_child(root)

    def episode_end(self, reward):
        pass

    # function for node traversal
    def traverse(self, node):
        while self.fully_expanded(node):
            node = self.best_uct(node)

        # in case no children are present / node is terminal
        return self.pick_univisted(node) or node

    # function for the result of the simulation
    def rollout(self, node):
        while self.non_terminal(node):
            node = self.rollout_policy(node)
        return self.result(node)

    # function for backpropagation
    def backpropagate(self, node, result):
        if self.is_root(node): return
        node.stats = self.update_stats(node, result)
        self.backpropagate(node.parent)

    # function for selecting the best child
    # node with highest number of visits
    def best_child(self, node):
        #pick child with highest number of visits
        pass

    def fully_expanded(self, node):
        raise NotImplementedError()

    def best_uct(self, node):
        raise NotImplementedError()

    def pick_univisted_child(self, node):
        raise NotImplementedError()

    def non_terminal(self, node):
        raise NotImplementedError()

    def rollout_policy(self, node):
        raise NotImplementedError()

    def result(self, node):
        raise NotImplementedError()

    def is_root(self, node):
        raise NotImplementedError()

    def update_stats(self, node):
        raise NotImplementedError()
