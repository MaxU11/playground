# The UCB MCTS agent

from .ucb_mcts_agent import UcbMCTSAgent
from .abstract_mcts_agent import Node
from .env_simulator import EnvSimulator
from .. import constants


class UcbLimitMCTSAgent(UcbMCTSAgent):
    """The Base-MCTS Agent."""

    def __init__(self, *args, **kwargs):
        super(UcbLimitMCTSAgent, self).__init__(*args, **kwargs)
        # parent hyperparameter
        self.expandTreeRollout = kwargs.get('expandTreeRollout', False)
        self.maxIterations = kwargs.get('maxIterations', 1000)
        self.maxTime = kwargs.get('maxTime', 0.1)
        self.discountFactor = kwargs.get('discountFactor', 0.9999)
        self.depthLimit = kwargs.get('depthLimit', 26)
        self.C = kwargs.get('C', 0.5)  # exploration weight

    def create_node(self, depth, game_data, action_space, agent_id, enemy_id, action, save_data=True):
        node = UcbMCTSAgent.create_node(self, depth, game_data, action_space, agent_id, enemy_id, action, save_data)

        node.unseen_actions = self.get_valid_actions(game_data, agent_id, node.unseen_actions)
        return node

    def get_valid_actions(self, game_data, agent_id, actions):
        # check valid actions
        valid_actions = []
        for agent in game_data.agents:
            if agent.agent_id is agent_id:
                if agent.is_alive:
                    valid_actions = EnvSimulator.get_valid_actions(game_data.board, game_data.flames, game_data.bombs, agent, actions)
                else:
                    valid_actions.append(constants.Action.Stop.value)
                if len(valid_actions) == 0:
                    #print(agent_id, 'good bye')
                    valid_actions.append(constants.Action.Stop.value)
                return valid_actions

        raise ValueError('Invalid agent id')