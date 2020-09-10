'''The base MCTS agent'''
import copy
import numpy as np

from abc import ABC, abstractmethod
from . import BaseAgent
from .env_simulator import Env_simulator

class AbstractMCTSAgent(ABC, BaseAgent):
    """The Base-MCTS Agent."""

    def __init__(self, *args, **kwargs):
        super(AbstractMCTSAgent, self).__init__(*args, **kwargs)

    def act(self, obs, action_space):
        root = self.get_root(obs, action_space)
        while self.is_search_active():
            leaf = self.traverse(root)
            simulation_result, leaf = self.rollout(leaf)
            self.backpropagate(leaf, simulation_result)

        self.search_finished()

        a = self.best_child(root)
        print(f'selected action: {a}')
        return self.best_child(root)

    # function for node traversal
    def traverse(self, node):
        while node.fully_expanded():
            node = self.get_selected_child(node)

        # in case no children are present / node is terminal
        return self.expand_node(node) or node

    # function for the result of the simulation
    def rollout(self, node):
        leaf = node
        while self.non_terminal(node):
            node = self.rollout_policy(node)
        if len(leaf.children) > 0:
            leaf = node
        return self.result(node), leaf

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

    @abstractmethod
    def search_finished(self):
        # called when search has finished
        pass


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
        self.agent_id = agent_id
        self.enemy_id = enemy_id
        self.action = action
        self.reward = 0

    def is_root(self):
        return self.parent == None

    def fully_expanded(self):
        return len(self.unseen_actions) <= 0

    def expand(self, action, game_state):
        child = Node(game_state, self.action_space, self.enemy_id, self.agent_id, action)
        child.parent = self
        self.children[action] = child
        self.unseen_actions.remove(action)
        return child

    def get_child(self, actions):
        child = self.children.setdefault(actions[self.agent_id], None)
        if child is not None:
            child = child.children.setdefault(actions[self.enemy_id], None)
        return child


class BaseMCTSAgent(AbstractMCTSAgent):
    """The Base-MCTS Agent. Manages the tree (adding nodes...)"""

    root = None

    def __init__(self, *args, **kwargs):
        super(BaseMCTSAgent, self).__init__(*args, **kwargs)

    def get_root(self, obs, action_space):
        if (self.root == None):
            game_state = Env_simulator.get_initial_game_state(obs, self.agent_id)
            self.root = Node(game_state, action_space, self.agent_id, self.enemies[0].value - 10, None)
        else:
            actions, reset = Env_simulator.update(self.root.game_state, obs, self.agent_id)
            child = self.root.get_child(actions)
            if not reset and child != None and Env_simulator.boards_equal(self.root.game_state.board, child.game_state.board, False):
                child.game_state = self.root.game_state
                self.root = child
            else:
                #print(f'ROOT RESET!\n{actions}\n')
                #print(f'{child}\n{self.root.game_state.board}\n{child.game_state.board}')
                self.root = Node(self.root.game_state, action_space, self.agent_id, self.enemies[0].value - 10, None)

        self.root_changed(self.root)
        return self.root

    @abstractmethod
    def root_changed(self, root):
        # signal that root has changed
        raise NotImplementedError()

    def expand_node(self, node):
        # pick unvisited child
        if node.agent_id == self.agent_id:
            action = self.get_my_expand_action(node)
            node = node.expand(action, node.game_state)

        game_state = copy.deepcopy(node.game_state)
        action = self.get_enemy_expand_action(node)
        Env_simulator.act(game_state, {node.enemy_id: node.action, node.agent_id: action})
        return node.expand(action, game_state)

    @abstractmethod
    def get_my_expand_action(self, node):
        # return action from my agent
        raise NotImplementedError()

    @abstractmethod
    def get_enemy_expand_action(self, node):
        # return action from my agent
        raise NotImplementedError()

    def non_terminal(self, node):
        # check if node is terminal
        return not node.game_state.done

    def rollout_policy(self, node):
        expand_tree = True
        # get next node based on rollout policy
        actions = {}
        if node.agent_id != self.agent_id:
            actions[self.agent_id] = node.action
        else:
            actions[self.agent_id] = self.get_my_rollout_action(node)
            if expand_tree: node = node.expand(actions[self.agent_id], node.game_state)

        actions[self.enemies[0].value - 10] = self.get_enemy_rollout_action(node)

        game_state = copy.deepcopy(node.game_state)
        Env_simulator.act(game_state, actions)

        if expand_tree: new_node = node.expand(actions[self.enemies[0].value - 10], game_state)
        else: new_node = Node(game_state, node.action_space, self.agent_id, self.enemies[0].value - 10, None)

        return new_node

    @abstractmethod
    def get_my_rollout_action(self, node):
        # return action from my agent
        raise NotImplementedError()

    @abstractmethod
    def get_enemy_rollout_action(self, node):
        # return action from my agent
        raise NotImplementedError()

    def search_finished(self):
        # called when search has finished
        self.save_tree_info(f'c:\\tmp\\{self.root.game_state.step_count}.txt', self.root)
        pass

    def get_tree_info(self, root):
        step_count = root.game_state.step_count
        if root.agent_id != self.agent_id:
            raise ValueError('Invalid root object')

        map_id = ''
        info = "{0:0=3d}\n".format(step_count)
        info += f'root {self.get_agent_map_info(self.root)}\n'
        info += np.array_str(self.root.game_state.board) + '\n'
        i, b = self.get_map_info(map_id, self.root, '')
        info += i + '\n' + b
        return info

    def get_map_info(self, map_id, node, intend):
        info = ''
        board_info = ''
        child_id = map_id + '-'

        child_actions = node.children.keys()
        for my_action in range(6):
            if my_action in child_actions:
                my_child = node.children[my_action]
                sub_child_id = f'{child_id}{my_action}'
                info += f'{intend}id: {sub_child_id}, {self.get_agent_map_info(my_child)}\n'

                enemy_child_actions = my_child.children.keys()
                for enemy_action in range(6):
                    if enemy_action in enemy_child_actions:
                        enemy_child = my_child.children[enemy_action]
                        subsub_child_id = f'{sub_child_id}{enemy_action}'
                        info += f'{intend}  id: {subsub_child_id}, {self.get_agent_map_info(enemy_child)}\n'
                        board_info += f'id: {subsub_child_id}\n{np.array_str(enemy_child.game_state.board)}\n'
                        i, b = self.get_map_info(subsub_child_id, enemy_child, intend + '    ')
                        info += i
                        board_info += b
        return info, board_info

    @abstractmethod
    def get_agent_map_info(self, node):
        # return action from my agent
        return ''

    def save_tree_info(self, path, root):
        info = self.get_tree_info(root)
        with open(path, 'w') as f:
            f.write(info)