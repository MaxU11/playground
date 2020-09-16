# The UCB MCTS agent
import math
import random

from .. import utility
from collections import defaultdict
from .ucb_rnd_mcts_agent import UBC_RND_MCTSAgent
from .base_mcts_agent import Node
from .env_simulator import Env_simulator
from .env_simulator import Game_state
from .. import constants


class UBC_RND_Limit_MCTSAgent(UBC_RND_MCTSAgent):
    """The Base-MCTS Agent."""

    root = None

    def __init__(self, *args, **kwargs):
        super(UBC_RND_Limit_MCTSAgent, self).__init__(*args, **kwargs)
        self.maxIterations = 1000
        self.iterations = 0
        self.discount_factor = 0.9999

    def create_node(self, game_state, action_space, agent_id, enemy_id, action):
        node = Node(game_state, action_space, agent_id, enemy_id, action)

        # check valid actions
        valid_actions = []
        for agent in node.game_state.agents:
            if agent.agent_id is agent_id:
                if agent.is_alive:
                    for action in node.unseen_actions:
                        if Env_simulator.is_valid_action(game_state.board, game_state.flames, game_state.bombs, agent, action):
                            valid_actions.append(action)
                else:
                    valid_actions.append(constants.Action.Stop.value)
                node.unseen_actions = valid_actions

                #print(node.game_state.board)
                #print(agent_id, valid_actions)
                return node

        raise ValueError('Invalid agent id')