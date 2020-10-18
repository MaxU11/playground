'''The base MCTS agent'''

import numpy as np
import datetime

from abc import abstractmethod
from .abstract_mcts_skeleton import AbstractMCTSSkeleton
from .env_simulator import EnvSimulator


class AbstractMCTSAgent(AbstractMCTSSkeleton):
    """The Base-MCTS Agent. Manages the tree (adding nodes...)"""

    root = None

    def __init__(self, *args, **kwargs):
        super(AbstractMCTSAgent, self).__init__(*args, **kwargs)
        # hyperparameter
        self.expandTreeRollout = kwargs.get('expandTreeRollout', False)
        self.maxIterations = kwargs.get('maxIterations', 1000)
        self.maxTime = kwargs.get('maxTime', 0.1)

        self.iterations = 0
        self.startTime = 0
        self.file_step_count = 0

    def agent_reset(self):
        self.root = None
        self.iterations = 0
        self.startTime = 0
        self.file_step_count = 0

    def get_root(self, obs, action_space):
        self.iterations = 0
        if self.maxTime > 0.0:
            self.startTime = datetime.datetime.now()

        if (self.root == None):
            game_data = EnvSimulator.get_initial_game_data(obs, self.agent_id)
            self.root = self.create_node(0, game_data, action_space, self.agent_id, self.enemies[0].value - 10, None)
        else:
            game_data = Node.get_game_data(self.root.game_state)
            game_data, actions, reset = EnvSimulator.update(game_data, obs, self.agent_id)
            child = self.root.get_child(actions)
            if not reset and child and EnvSimulator.boards_equal(game_data.board, Node.get_game_data(child.game_state).board, False):
                child.game_state, done = Node.get_game_state(game_data)
                self.root = child
                self.root.parent = None
                self.root.done = done
            else:
                self.root = self.create_node(0, game_data, action_space, self.agent_id, self.enemies[0].value - 10, None)

        self.root_changed(self.root)
        return self.root

    @abstractmethod
    def root_changed(self, root):
        # signal that root has changed
        raise NotImplementedError()

    def is_search_active(self):
        self.iterations += 1
        if self.maxTime > 0.0:
            timeDiff = datetime.datetime.now() - self.startTime
            return timeDiff.total_seconds() < self.maxTime
        else:
            return self.iterations <= self.maxIterations

    def get_data(self, node):
        return Node.get_game_data(node.game_state)

    def expand_node(self, node):
        game_data = Node.get_game_data(node.game_state)

        # pick unvisited child
        if node.agent_id == self.agent_id:
            action = self.get_my_expand_action(node)
            node = node.expand(action, game_data, self)

        action = self.get_enemy_expand_action(node)
        EnvSimulator.act(game_data, {node.enemy_id: node.action, node.agent_id: action})
        return node.expand(action, game_data, self)

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
        return not node.done

    def rollout_policy(self, node, data):
        # get next node based on rollout policy
        actions = {}
        if node.agent_id != self.agent_id:
            actions[self.agent_id] = node.action
        else:
            actions[self.agent_id] = self.get_my_rollout_action(node, data)
            if self.expandTreeRollout: node = node.expand(actions[self.agent_id], data, self)
            else: node = self.create_node(node.depth + 1, data, node.action_space, self.enemies[0].value - 10, self.agent_id, actions[self.agent_id], False)

        actions[self.enemies[0].value - 10] = self.get_enemy_rollout_action(node, data)

        EnvSimulator.act(data, actions)
        #print(actions, '\n', data.board)

        if self.expandTreeRollout: new_node = node.expand(actions[self.enemies[0].value - 10], data, self)
        else: new_node = self.create_node(node.depth + 1, data, node.action_space, self.agent_id, self.enemies[0].value - 10, actions[self.enemies[0].value - 10], False)

        return new_node, data

    @abstractmethod
    def get_my_rollout_action(self, node, data):
        # return action from my agent
        raise NotImplementedError()

    @abstractmethod
    def get_enemy_rollout_action(self, node, data):
        # return action from my agent
        raise NotImplementedError()

    def create_node(self, depth, game_data, action_space, agent_id, enemy_id, action, save_data=True):
        # create the node
        if save_data:
            return Node(depth, game_data, action_space, agent_id, enemy_id, action)
        else:
            new_node = Node(depth, None, action_space, agent_id, enemy_id, action)
            new_node.done = EnvSimulator.get_done(game_data)
            return new_node

    def search_finished(self):
        self.file_step_count += 1
        # called when search has finished
        self.save_tree_info(f'c:\\tmp\\{self.agent_id}_{self.file_step_count}.txt', self.root)
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

class Node():

    def __init__(self, depth, game_data, action_space, agent_id, enemy_id, action, valid_actions=None):
        if not game_data is None:
            self.game_state, self.done = Node.get_game_state(game_data)
        else:
            self.game_state = None
            self.done = False
        self.parent = None
        self.children = {}
        self.action_space = action_space
        if valid_actions is None:
            self.unseen_actions = list(range(self.action_space.n))
        else:
            self.unseen_actions = valid_actions
        self.agent_id = agent_id
        self.enemy_id = enemy_id
        self.action = action
        self.reward = 0
        self.depth = depth

    def is_root(self):
        return self.parent == None

    def fully_expanded(self):
        return len(self.unseen_actions) <= 0

    def expand(self, action, game_data, agent):
        child = agent.create_node(self.depth + 1, game_data, self.action_space, self.enemy_id,
                                  self.agent_id, action)
        child.parent = self
        self.children[action] = child
        self.unseen_actions.remove(action)
        return child

    def get_child(self, actions):
        child = self.children.setdefault(actions[self.agent_id], None)
        if child is not None:
            child = child.children.setdefault(actions[self.enemy_id], None)
        return child

    @staticmethod
    def get_game_state(game_data):
        return EnvSimulator.get_game_state(game_data)

    @staticmethod
    def get_game_data(game_state):
        return EnvSimulator.get_game_data(game_state)