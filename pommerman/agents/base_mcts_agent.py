'''The base MCTS agent'''
from abc import ABC, abstractmethod
from . import BaseAgent

class BaseMCTSAgent(ABC, BaseAgent):
    """The Base-MCTS Agent."""

    def __init__(self, *args, **kwargs):
        super(BaseMCTSAgent, self).__init__(*args, **kwargs)

    def act(self, obs, action_space):
        leaf = self.get_root(obs, action_space)
        while self.is_search_active():
            leaf = self.traverse(leaf)
            simulation_result, leaf = self.rollout(leaf)
            self.backpropagate(leaf, simulation_result)

        return self.best_child(obs)

    # function for node traversal
    def traverse(self, node):
        while node.fully_expanded():
            node = self.select_child(node)

        # in case no children are present / node is terminal
        return self.expand_node(node) or node

    # function for the result of the simulation
    def rollout(self, node):
        leaf = node
        while self.non_terminal(node):
            node = self.rollout_policy(node)
        if len(leaf.children) > 0: leaf = node
        return self.result(node), leaf

    # function for backpropagation
    def backpropagate(self, node, result):
        if node.is_root():
            return
        node.stats = self.update_stats(node, result)
        self.backpropagate(node.parent, result)

    @abstractmethod
    def get_root(self, obs, action_space):
        # initialize root node
        raise NotImplementedError()

    @abstractmethod
    def is_search_active(self):
        # is search active?
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
    def update_stats(self, node, result):
        # get updated node stats
        raise NotImplementedError()

    @abstractmethod
    def best_child(self, node):
        # pick child with highest number of visits
        raise NotImplementedError()


    def episode_end(self, reward):
        """This is called at the end of the episode to let the agent know that
        the episode has ended and what is the reward.

        Args:
          reward: The single reward scalar to this agent.
        """
        pass

class Node():

    def __init__(self, game_state, action_space, agent_id, enemy_id, action):
        self.game_state = game_state
        self.parent = None
        self.children = {}
        self.action_space = action_space
        self.unseen_actions = list(range(self.action_space.n))
        self.state = None
        self.agent_id = agent_id
        self.enemy_id = enemy_id
        self.action = action

    def is_root(self):
        return self.parent == None

    def fully_expanded(self):
        return len(self.unseen_actions) <= 0

    def expand(self, action, game_state):
        child = Node(game_state, self.action_space, self.enemy_id, self.agent_id, action)
        self.children[action] = child
        self.unseen_actions.remove(action)
        return child

    def get_child(self, actions):
        child = self.children[actions[self.agent_id]]
        if child is not None:
            child = child.children[actions[self.enemy_id]]
        return child