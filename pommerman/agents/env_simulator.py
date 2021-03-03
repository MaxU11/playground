import numpy as np
import _pickle as cPickle
import copy

from pommerman import forward_model
from pommerman import constants
from pommerman import characters
from pommerman import utility


class EnvSimulator:

    @staticmethod
    def get_initial_game_data(obs, my_id, max_steps=1000):
        game_data = GameData()
        game_data.board_size = len(obs['board'])
        game_data.step_count = obs['step_count'] - 1
        game_data.max_steps = max_steps
        game_data.game_type = obs['game_type']
        game_data.simulation_bomb_life = None

        # board
        game_data.board = EnvSimulator.get_board(game_data.board_size, obs['board'])

        # items
        game_data.items = {}

        # agents
        game_data.agents = []
        for id in [constants.Item.Agent0.value - 10, constants.Item.Agent1.value - 10]:
            board_id = id + 10
            agent = characters.Bomber(id, game_data.game_type)
            agent.set_start_position(EnvSimulator.get_position(game_data.board, board_id, True))
            if (id == my_id):
                agent.reset(int(obs['ammo']), board_id in obs['alive'], int(obs['blast_strength']),
                            bool(obs['can_kick']))
            else:
                agent.reset(agent.ammo, board_id in obs['alive'], agent.blast_strength, agent.can_kick)
            game_data.agents.append(agent)

        # bombs
        game_data.bombs = []
        bomb_array = EnvSimulator.get_position(game_data.board, constants.Item.Bomb.value, False)
        if len(bomb_array) > 0:
            raise ValueError('Invalid: no bombs allowed in initial state')

        # flames
        game_data.flames = []
        flame_array = EnvSimulator.get_position(game_data.board, constants.Item.Flames.value, False)
        if len(flame_array) > 0:
            raise ValueError('Invalid: no flames allowed in initial state')

        # done
        game_data.done = forward_model.ForwardModel.get_done(game_data.agents, game_data.step_count,
                                                             game_data.max_steps, game_data.game_type, None)

        return game_data

    @staticmethod
    def update(game_data, obs, my_id):
        enemy_id = 0
        if my_id is 0: enemy_id = 1

        if (game_data.board_size != len(obs['board'])):
            raise ValueError('Invalid update: boardsize different!')
        if (game_data.step_count + 1 != obs['step_count']):
            raise ValueError('Invalid update: missed step count!')
        game_data.step_count = obs['step_count']

        new_board = EnvSimulator.get_board(game_data.board_size, obs['board'])
        new_bomb_life = EnvSimulator.get_board(game_data.board_size, obs['bomb_life'], 0)
        new_bomb_strength = EnvSimulator.get_board(game_data.board_size, obs['bomb_blast_strength'], 0)

        reset = False

        # get actions
        actions = {}
        for a in game_data.agents:
            old_pos = EnvSimulator.get_position(game_data.board, a.agent_id + 10, True)
            new_pos = EnvSimulator.get_position(new_board, a.agent_id + 10, True)

            if not a.is_alive:
                raise ValueError('update error: agent life!')

            for b in game_data.bombs:
                if b.moving_direction != None:
                    pass

            if (old_pos != new_pos):
                actions[a.agent_id] = utility.get_direction(old_pos, new_pos).value
                if not a.can_kick and game_data.board[new_pos] == constants.Item.Bomb.value:
                    for b in game_data.bombs:
                        if b.position == new_pos and b.moving_direction != None:
                            a.can_kick = True
                            reset = True
            elif new_bomb_life[new_pos] == constants.DEFAULT_BOMB_LIFE:
                actions[a.agent_id] = constants.Action.Bomb.value
                if a.ammo == 0:
                    a.ammo += 1
                    reset = True
                if a.blast_strength != new_bomb_strength[new_pos]:
                    a.blast_strength = new_bomb_strength[new_pos]
                    reset = True
            else:
                actions[a.agent_id] = constants.Action.Stop.value

        save_game_data = copy.deepcopy(game_data)
        EnvSimulator.act(game_data, actions)

        if game_data.agents[0].is_alive != (10 in obs['alive']):
            raise ValueError(f'update error: agent life!\n\n{game_data.board}\n\n{new_board}')
        if game_data.agents[1].is_alive != (11 in obs['alive']):
            raise ValueError(f'update error: agent life!\n\n{game_data.board}\n\n{new_board}')
        if (len(game_data.bombs) != len(new_bomb_life[new_bomb_life > 0])):
            raise ValueError(f'update error: bomb count!\n\n{game_data.board}\n\n{new_board}')

        # print("board: \n", game_data.board)
        # print("agent1: ", game_data.agents[0].ammo, game_data.agents[0].blast_strength, game_data.agents[0].can_kick)
        # print("agent2: ", game_data.agents[1].ammo, game_data.agents[1].blast_strength, game_data.agents[1].can_kick)

        # compare boards
        equal, equal_noitems = EnvSimulator.boards_equal(game_data.board, new_board, True)
        if not equal:
            if equal_noitems:
                reset = True  # EQUAL WITHOUT ITEMS => SOMEWHERE NEW ITEMS AVAILABLE -> RESET
            else:
                print('board unequal: {game_data.board}\n\n{new_board}\n\n{actions}')
                def find_actions(save_game_data, actions):
                    actions_1 = [actions[0]] if actions[0] != 0 else range(1, 6)
                    actions_2 = [actions[1]] if actions[1] != 0 else range(1, 6)
                    for a1 in actions_1:
                        for a2 in actions_2:
                            game_data = copy.deepcopy(save_game_data)
                            acts = {0: a1, 1: a2}
                            EnvSimulator.act(game_data, acts)
                            eq, eq_noitems = EnvSimulator.boards_equal(game_data.board, new_board, True)
                            if eq_noitems:
                                return game_data, acts, eq
                    return None, None, False

                game_data, actions, eq = find_actions(save_game_data, actions)
                print(f'found game_data: {game_data}\n\n{actions}')
                if not game_data:
                    raise ValueError(f'should not happen anymore')
                if not eq:
                    reset = True # EQUAL WITHOUT ITEMS => SOMEWHERE NEW ITEMS AVAILABLE -> RESET

        game_data.agents[my_id].ammo = int(obs['ammo'])
        game_data.agents[my_id].blast_strength = int(obs['blast_strength'])
        game_data.agents[my_id].can_kick = bool(obs['can_kick'])

        # update board because of items
        game_data.board = new_board

        return game_data, actions, reset

    @staticmethod
    def act(game_data, actions):

        if game_data.simulation_bomb_life:
            for b in game_data.bombs:
                if b.life > game_data.simulation_bomb_life: b.life = game_data.simulation_bomb_life

        game_data.board, \
        game_data.agents, \
        game_data.bombs, \
        game_data.items, \
        game_data.flames = forward_model.ForwardModel.step(actions,
                                                            game_data.board,
                                                            game_data.agents,
                                                            game_data.bombs,
                                                            game_data.items,
                                                            game_data.flames)

        if game_data.simulation_bomb_life:
            for b in game_data.bombs:
                if b.life > 2: b.life = 2

        # done
        game_data.done = forward_model.ForwardModel.get_done(game_data.agents, game_data.step_count,
                                                              game_data.max_steps, game_data.game_type, None)

    @staticmethod
    def get_done(game_data):
        return game_data.done

    @staticmethod
    def get_alive(game_data):
        alive = {}
        for a in game_data.agents:
            alive[a.agent_id] = a.is_alive
        return alive

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
                raise ValueError("Invalid pos count!", board, item)
            return pos[0]
        else:
            return pos

    @staticmethod
    def get_valid_actions(board, flames, bombs, agent, actions):
        valid_actions = []
        invalid_values = None
        invalid_positions = None
        row, col = agent.position

        for action in actions:
            if action is constants.Action.Bomb.value:
                if agent.ammo > 0: valid_actions.append(action)
            else:
                if invalid_values is None:
                    invalid_values = [item.value for item in [constants.Item.Rigid, constants.Item.Wood]]
                    if not agent.can_kick: invalid_values.append(constants.Item.Bomb.value)
                if invalid_positions is None:
                    invalid_positions = EnvSimulator.get_invalid_positions(board, flames, bombs)
                if EnvSimulator.is_valid_direction(board, row, col, action, invalid_values, invalid_positions):
                    valid_actions.append(action)
        return valid_actions

    @staticmethod
    def boards_equal(board1, board2, ignore_items):
        comparison = (board1 == board2).all()

        if ignore_items:
            board1 = copy.deepcopy(board1)
            board2 = copy.deepcopy(board2)
            board1[board1 == constants.Item.ExtraBomb.value] = constants.Item.Passage.value
            board1[board1 == constants.Item.IncrRange.value] = constants.Item.Passage.value
            board1[board1 == constants.Item.Kick.value] = constants.Item.Passage.value
            board2[board2 == constants.Item.ExtraBomb.value] = constants.Item.Passage.value
            board2[board2 == constants.Item.IncrRange.value] = constants.Item.Passage.value
            board2[board2 == constants.Item.Kick.value] = constants.Item.Passage.value
            comparison_ignore = (board1 == board2).all()
            return comparison, comparison_ignore

        return comparison.all()

    @staticmethod
    def boards_equal_speed(board1, board2, ignore_items):
        comparison = (board1 != board2)

        if ignore_items:
            diff_items = False
            b1_diff = board1[comparison]
            b2_diff = board2[comparison]
            b1_no_items = (b1_diff != constants.Item.ExtraBomb.value) & \
                          (b1_diff != constants.Item.IncrRange.value) & \
                          (b1_diff != constants.Item.Kick.value) & \
                          (b1_diff != constants.Item.Passage.value)
            diff_items = b1_no_items.any()
            if not diff_items:
                b2_no_items = (b2_diff != constants.Item.ExtraBomb.value) & \
                              (b2_diff != constants.Item.IncrRange.value) & \
                              (b2_diff != constants.Item.Kick.value) &\
                              (b2_diff != constants.Item.Passage.value)
                diff_items = b2_no_items.any()
            return diff_items

        return not comparison.any()

    @staticmethod
    def get_boards_differences(board1, board2):
        board1 = copy.deepcopy(board1)
        board2 = copy.deepcopy(board2)
        board1[board1 == constants.Item.ExtraBomb.value] = constants.Item.Passage.value
        board1[board1 == constants.Item.IncrRange.value] = constants.Item.Passage.value
        board1[board1 == constants.Item.Kick.value] = constants.Item.Passage.value
        board2[board2 == constants.Item.ExtraBomb.value] = constants.Item.Passage.value
        board2[board2 == constants.Item.IncrRange.value] = constants.Item.Passage.value
        board2[board2 == constants.Item.Kick.value] = constants.Item.Passage.value

        a1bomb = a2bomb = kick = flame = False
        comparison = (board1 == board2)
        diffs = np.where(comparison is False)
        if len(diffs) >= 2:
            diffs = list(zip(diffs[0], diffs[1]))
            for diff in diffs:
                prev_item = board1[diff]
                new_item = board2[diff]
                if prev_item == constants.Item.Agent1.value and new_item == constants.Item.Bomb.value:
                    a1bomb = True
                elif prev_item == constants.Item.Agent2.value and new_item == constants.Item.Bomb.value:
                    a2bomb = True
                elif prev_item == constants.Item.Passage.value and new_item == constants.Item.Bomb.value:
                    kick = True
                elif new_item == constants.Item.Flames.value:
                    flame = True
                else:
                    raise ValueError('Invalid difference between maps.')
        # else:
            # print(comparison, "diffs: ", diffs)

        return a1bomb, a2bomb, kick, flame


    @staticmethod
    def get_invalid_positions(board, flames, bombs):
        invalid_positions = []
        for flame in flames:
            if flame.life > 0:
                invalid_positions.append(flame.position)

        exploded = True
        exp_bombs = []
        while exploded:
            exploded = False
            for bomb in bombs:
                if bomb not in exp_bombs and (bomb.life is 1 or bomb.position in invalid_positions):
                    EnvSimulator._get_bomb_fire_positions(board, bomb, invalid_positions)
                    exp_bombs.append(bomb)
                    exploded = True

        return invalid_positions

    @staticmethod
    def is_valid_direction(board, row, col, direction, invalid_values, invalid_positions):
        '''Determins if a move is in a valid direction'''
        if constants.Action(direction) == constants.Action.Up:
            return row - 1 >= 0 and board[row - 1][col] not in invalid_values and (
            row - 1, col) not in invalid_positions
        elif constants.Action(direction) == constants.Action.Down:
            return row + 1 < len(board) and board[row + 1][col] not in invalid_values and (
            row + 1, col) not in invalid_positions
        elif constants.Action(direction) == constants.Action.Left:
            return col - 1 >= 0 and board[row][col - 1] not in invalid_values and (
            row, col - 1) not in invalid_positions
        elif constants.Action(direction) == constants.Action.Right:
            return col + 1 < len(board[0]) and board[row][col + 1] not in invalid_values and (
            row, col + 1) not in invalid_positions
        elif constants.Action(direction) == constants.Action.Stop:
            return board[row][col] not in invalid_values and (
                row, col) not in invalid_positions

        raise constants.InvalidAction("We did not receive a valid direction: ", direction)

    @staticmethod
    def _get_bomb_fire_positions(board, bomb, fire_pos):
        fire_pos.append(bomb.position)
        EnvSimulator._get_fire_positions_in_direction(board, bomb.position[0], bomb.position[1],
                                                       bomb.blast_strength - 1, 0, 1, fire_pos)  # right
        EnvSimulator._get_fire_positions_in_direction(board, bomb.position[0], bomb.position[1],
                                                       bomb.blast_strength - 1, 0, -1, fire_pos)  # left
        EnvSimulator._get_fire_positions_in_direction(board, bomb.position[0], bomb.position[1],
                                                       bomb.blast_strength - 1, -1, 0, fire_pos)  # up
        EnvSimulator._get_fire_positions_in_direction(board, bomb.position[0], bomb.position[1],
                                                       bomb.blast_strength - 1, 1, 0, fire_pos)  # down

    @staticmethod
    def _get_fire_positions_in_direction(board, x, y, strength, x_dir, y_dir, fire_pos):
        if strength <= 0 or not utility.position_on_board(board, (x, y)):
            return
        next_x = x + x_dir
        next_y = y + y_dir
        if not utility.position_on_board(board, (next_x, next_y)):
            return
        if utility.position_in_items(board, (next_x, next_y), [constants.Item.Rigid, constants.Item.Wood]):
            return

        fire_pos.append((next_x, next_y))
        EnvSimulator._get_fire_positions_in_direction(board, next_x, next_y, strength - 1, x_dir, y_dir, fire_pos)

    @staticmethod
    def get_bomb_items(board, bomb_pos, bomb_strength):
        items = []
        EnvSimulator._get_items_in_direction(board, bomb_pos, bomb_strength - 1, 0, 1, items)
        EnvSimulator._get_items_in_direction(board, bomb_pos, bomb_strength - 1, 0, -1, items)
        EnvSimulator._get_items_in_direction(board, bomb_pos, bomb_strength - 1, -1, 0, items)
        EnvSimulator._get_items_in_direction(board, bomb_pos, bomb_strength - 1, 1, 0, items)
        return items

    @staticmethod
    def _get_items_in_direction(board, pos, strength, x_dir, y_dir, items):
        if strength <= 0 or not utility.position_on_board(board, pos):
            return
        x, y = pos
        next_x = x + x_dir
        next_y = y + y_dir
        if not utility.position_on_board(board, (next_x, next_y)):
            return
        item = board[(next_x, next_y)]
        try:
            if type(item) == tuple:
                print(item)
            if not item in items:
                items.append(item)
        except:
            if type(item) == tuple:
                print(item)
            print(item, items)
        if utility.position_in_items(board, (next_x, next_y), [constants.Item.Rigid, constants.Item.Wood]):
            return
        EnvSimulator._get_fire_positions_in_direction(board, next_x, next_y, strength - 1, x_dir, y_dir, items)

    @staticmethod
    def get_game_state(game_data):
        # return game_data, EnvSimulator.get_done(game_data)
        return cPickle.dumps(game_data), EnvSimulator.get_done(game_data)

    @staticmethod
    def get_game_data(game_state):
        # return copy.deepcopy(game_state)
        return cPickle.loads(game_state)

class GameData:
    pass
