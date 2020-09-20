
from . import SimpleAgent
from .. import constants


class Simulate_SimpleAgent(SimpleAgent):

    def __init__(self, *args, **kwargs):
        super(SimpleAgent, self).__init__(*args, **kwargs)

    def act(self, obs, action_space):
        # obs['alive'] =
        # obs['board'] = board
        # obs['bomb_blast_strength'] =
        # obs['bomb_life'] =
        # obs['bomb_moving_direction'] =
        # obs['flame_life'] =
        # obs['game_type'] =
        # obs['game_env'] =
        # obs['position'] =
        # obs['blast_strength'] =
        # obs['can_kick'] =
        # obs['teammate'] =
        # obs['ammo'] =
        # obs['enemies'] =
        # obs['step_count'] =

        obs['board'] = [[ 0,  0,  2,  1,  1,  1],
                         [ 6,  0, 10,  0,  0,  0],
                         [ 2,  0,  0,  1,  0,  1],
                         [ 1,  3,  1,  0,  0,  1],
                         [ 1,  0,  0,  0,  0,  1],
                         [ 1, 11,  1,  1,  1,  0]]
        obs['bomb_blast_strength'] = [[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 2, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]]
        obs['bomb_life'] = [[0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0]]
        obs['position'] = (5, 1)
        obs['enemies'][0] = constants.Item.Agent0

        agent = SimpleAgent()
        action = agent.act(obs, action_space)

        print('Simple agent: ', action)

        action = agent.act(obs, action_space)
        return action