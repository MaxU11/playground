# The UCB MCTS agent

from .ucb_mcts_agent import UcbMCTSAgent
from .abstract_mcts_agent import Node
from .env_simulator import Env_simulator
from .. import constants


class UcbLimitMCTSAgent(UcbMCTSAgent):
    """The Base-MCTS Agent."""

    def __init__(self, *args, **kwargs):
        super(UcbLimitMCTSAgent, self).__init__(*args, **kwargs)
        self.maxIterations = 100
        self.iterations = 0
        self.discount_factor = 0.9999
        self.depth_limit = 26
        self.expand_tree_rollout = False

    def create_node(self, depth, game_data, action_space, agent_id, enemy_id, action, save_data=True):
        if save_data:
            node = Node(depth, game_data, action_space, agent_id, enemy_id, action)
        else:
            node = Node(depth, None, action_space, agent_id, enemy_id, action)

        # check valid actions
        valid_actions = []
        for agent in game_data.agents:
            if agent.agent_id is agent_id:
                if agent.is_alive:
                    for action in node.unseen_actions:
                        if Env_simulator.is_valid_action(game_data.board, game_data.flames, game_data.bombs, agent, action):
                            valid_actions.append(action)
                else:
                    valid_actions.append(constants.Action.Stop.value)
                node.unseen_actions = valid_actions

                return node

        raise ValueError('Invalid agent id')