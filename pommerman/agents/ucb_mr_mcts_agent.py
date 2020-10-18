# The UCB MiniMax-Rollout MCTS agent

import random
import _pickle as cPickle

from .ucb_mcts_agent import UcbMCTSAgent
from .env_simulator import EnvSimulator


class UcbMRMCTSAgent(UcbMCTSAgent):
    """The MiniMaxRollout-MCTS Agent."""

    def __init__(self, *args, **kwargs):
        super(UcbMRMCTSAgent, self).__init__(*args, **kwargs)
        # parent hyperparameter
        self.expandTreeRollout = kwargs.get('expandTreeRollout', False)
        self.maxIterations = kwargs.get('maxIterations', 1000)
        self.maxTime = kwargs.get('maxTime', 0.1)
        self.discountFactor = kwargs.get('discountFactor', 0.9999)
        self.depthLimit = kwargs.get('depthLimit', 26)
        self.C = kwargs.get('C', 0.5)  # exploration weight
        # hyperparameter
        self.MRDepthLimit = kwargs.get('MRDepthLimit', 2)

    def get_my_rollout_action(self, node, data):
        return self.get_rollout_action(node, data)

    def get_enemy_rollout_action(self, node, data):
        return self.get_rollout_action(node, data)

    def get_rollout_action(self, node, data):
        data_pickle = cPickle.dumps(data)
        possible_actions = []
        possible_reward = -99

        my_actions = self.getActionSpace(data, node.agent_id, node.unseen_actions)
        enemy_actions = self.getActionSpace(data, node.enemy_id, list(range(node.action_space.n)))
        for action in my_actions:
            minimax_value = self.getMinMaxValue(data_pickle, action, enemy_actions, node.agent_id, node.enemy_id, self.MRDepthLimit - 2)
            if minimax_value is 1: # win
                return action
            elif minimax_value is possible_reward:
                possible_actions.append(action)
            elif minimax_value > possible_reward:
                possible_reward = minimax_value
                possible_actions = [action]

        if possible_actions:
            return random.choice(possible_actions)
        else:
            return random.choice(node.unseen_actions)

    def getMinMaxValue(self, data_pickle, action, action_space, my_id, enemy_id, depth_limit):
        ret_win = 99

        for enemy_action in action_space:
            win = 0
            actions = {}
            actions[my_id] = action
            actions[enemy_id] = enemy_action

            data = cPickle.loads(data_pickle)
            EnvSimulator.act(data, actions)

            if EnvSimulator.get_done(data):
                win = self.getReward(data, my_id)
            elif depth_limit > 0:
                my_actions = self.getActionSpace(data, my_id, list(range(6)))
                enemy_actions = self.getActionSpace(data, enemy_id, list(range(6)))
                next_data_pickle = cPickle.dumps(data)
                win = -99
                for next_action in my_actions:
                    win = max(win, self.getMinMaxValue(next_data_pickle, next_action, action_space, my_id, enemy_id, depth_limit - 2))
                    if win is 1:  # win
                        break
            #else:
                # do a rollout
                #limit = -1
                #cur = 0
                #while not EnvSimulator.get_done(data) and cur <= limit:
                #    cur += 1
                #    actions[my_id] = random.choice(space_actions)
                #    actions[enemy_id] = random.choice(space_actions)
                #    EnvSimulator.act(data, actions)
                #win = self.getReward(data, my_id) * 0.8

            ret_win = min(ret_win, win)
            if win is -2:  # lose
                break

        return ret_win

    def getActionSpace(self, game_data, agent_id, action_space):
        return action_space

    def getReward(self, data, my_id):
        win = 0
        if EnvSimulator.get_done(data):
            for a in data.agents:
                if not a.is_alive:
                    if a.agent_id is my_id:
                        win -= 2
                    else:
                        win += 1
        return win
