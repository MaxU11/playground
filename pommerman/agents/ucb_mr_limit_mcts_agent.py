# The UCB MiniMax-Rollout MCTS agent

import random

from .ucb_mr_mcts_agent import UcbMRMCTSAgent
from .env_simulator import EnvSimulator
from .. import constants


class UcbMRLimitMCTSAgent(UcbMRMCTSAgent):
    """The LimitedMiniMaxRollout-MCTS Agent."""

    def __init__(self, *args, **kwargs):
        super(UcbMRMCTSAgent, self).__init__(*args, **kwargs)
        # parent hyperparameter
        self.expandTreeRollout = kwargs.get('expandTreeRollout', False)
        self.maxIterations = kwargs.get('maxIterations', 1000)
        self.maxTime = kwargs.get('maxTime', 0.1)
        self.discountFactor = kwargs.get('discountFactor', 0.9999)
        self.depthLimit = kwargs.get('depthLimit', 26)
        self.C = kwargs.get('C', 0.5)  # exploration weight
        self.MRDepthLimit = kwargs('MRDepthLimit', 2)

    def getActionSpace(self, game_data, agent_id, action_space):
        return self.get_valid_acions(game_data, agent_id, action_space)

    def get_valid_acions(self, game_data, agent_id, actions):
        # check valid actions
        valid_actions = []
        for agent in game_data.agents:
            if agent.agent_id is agent_id:
                if agent.is_alive:
                    valid_actions = EnvSimulator.get_valid_actions(game_data.board, game_data.flames, game_data.bombs, agent, actions)
                else:
                    valid_actions.append(constants.Action.Stop.value)
                return valid_actions

        raise ValueError('Invalid agent id')
