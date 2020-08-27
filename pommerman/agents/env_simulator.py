import numpy as np

from pommerman import forward_model
from pommerman import constants
from pommerman import characters
from pommerman import utility

class Env_simulator:

    @staticmethod
    def get_initial_game_state(obs, my_id, max_steps=1000):
        game_state = Game_state
        game_state.board_size = len(obs['board'])
        game_state.step_count = obs['step_count']
        game_state.max_steps = max_steps
        game_state.game_type = obs['game_type']

        # board
        game_state.board = Env_simulator.get_board(game_state.board_size, obs['board'])

        # items
        game_state.items = {}

        # agents
        game_state.agents = []
        for id in [constants.Item.Agent0.value, constants.Item.Agent1.value]:
            agent = characters.Bomber(id - 10, game_state.game_type)
            agent.set_start_position(Env_simulator.get_position(game_state.board, id, True))
            if (id == my_id):
                agent.reset(int(obs['ammo']), id in obs['alive'], int(obs['blast_strength']), bool(obs['can_kick']))
            else:
                agent.reset(agent.ammo, id in obs['alive'], agent.blast_strength, agent.can_kick)
            game_state.agents.append(agent)

        # bombs
        game_state.bombs = []
        bomb_array = Env_simulator.get_position(game_state.board, constants.Item.Bomb.value, False)
        if len(bomb_array) > 0:
            raise ValueError('Invalid: no bombs allowed in initial state')

        # flames
        game_state.flames = []
        flame_array = Env_simulator.get_position(game_state.board, constants.Item.Flames.value, False)
        if len(flame_array) > 0:
            raise ValueError('Invalid: no flames allowed in initial state')

        # done
        game_state.done = forward_model.ForwardModel.get_done(game_state.agents, game_state.step_count,
                                   game_state.max_steps, game_state.game_type, None)

        return game_state

    @staticmethod
    def update(game_state, obs):
        if (game_state.board_size != len(obs['board'])):
            raise ValueError('Invalid update: boardsize different!')
        if (game_state.step_count+1 != obs['step_count']):
            raise ValueError('Invalid update: missed step count!')
        game_state.step_count = obs['step_count']

        new_board = Env_simulator.get_board(game_state.board_size, obs['board'])
        new_bomb_life = Env_simulator.get_board(game_state.board_size, obs['bomb_life'], 0)

        # get actions
        actions = {}
        for a in game_state.agents:
            old_pos = Env_simulator.get_position(game_state.board, a.agent_id, True)
            new_pos = Env_simulator.get_position(new_board, a.agent_id, True)

            if (old_pos != new_pos):
                actions[a.agent_id] = utility.get_direction(old_pos, new_pos)
            elif new_bomb_life[new_pos] == constants.DEFAULT_BOMB_LIFE:
                actions[a.agent_id] = constants.Action.Bomb
            else:
                actions[a.agent_id] = constants.Action.Stop

        Env_simulator.act(game_state, actions)

        # compare boards
        if not Env_simulator.boards_equal(game_state.board, new_board, True):
            raise ValueError('Invalid board changes!')

        # update board because of items
        game_state.board = new_board

        return actions

    @staticmethod
    def act(game_state, actions):
        game_state.board, \
        game_state.agents, \
        game_state.bombs, \
        game_state.items, \
        game_state.flames = forward_model.ForwardModel.step(actions,
                                                            game_state.board,
                                                            game_state.agents,
                                                            game_state.bombs,
                                                            game_state.items,
                                                            game_state.flames)

        # done
        game_state.done = forward_model.ForwardModel.get_done(game_state.agents, game_state.step_count,
                                                 game_state.max_steps, game_state.game_type, None)

    @staticmethod
    def get_board(board_size, board_array, init_value=constants.Item.Passage.value):
        board = np.ones((board_size, board_size)).astype(np.uint8)
        board *= init_value
        for x in range(board_size):
            for y in range(board_size):
                board[x, y] = board_array[x][y]
        return board

    @staticmethod
    def get_position(board, item, is_single_pos):
        pos = np.where(board == item)
        pos = list(zip(pos[0], pos[1]))
        if is_single_pos:
            if len(pos) != 1:
                raise ValueError("Invalid pos count!")
            return pos[0]
        else:
            return pos

    @staticmethod
    def boards_equal(board1, board2, ignore_items):
        if ignore_items:
            board1[board1 == constants.Item.ExtraBomb.value] = constants.Item.Passage.value
            board1[board1 == constants.Item.IncrRange.value] = constants.Item.Passage.value
            board1[board1 == constants.Item.Kick.value] = constants.Item.Passage.value
            board2[board2 == constants.Item.ExtraBomb.value] = constants.Item.Passage.value
            board2[board2 == constants.Item.IncrRange.value] = constants.Item.Passage.value
            board2[board2 == constants.Item.Kick.value] = constants.Item.Passage.value

        comparison = (board1 == board2)
        return comparison.all()

class Game_state:
    pass

