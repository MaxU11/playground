import numpy as np
import copy

from pommerman import constants
from pommerman import utility

STEP_COUNT_POS = 0
DONE_POS = 1
AMMO_POS = 0
BLAST_STRENGTH_POS = 1
CAN_KICK_POS = 2
ALIVE_POS = 3
ROW_POS = 4
COL_POS = 5


class EnvSimulator:

    @staticmethod
    def get_initial_game_data(obs, my_id, max_steps=1000):
        board_size = len(obs['board'])

        game_data = EnvSimulator.get_board(board_size, obs['board'])
        agent_0_pos = EnvSimulator.get_position(game_data, 0, True)
        agent_1_pos = EnvSimulator.get_position(game_data, 1, True)

        game_info = np.zeros((1, board_size)).astype(np.uint16)
        game_info[0, STEP_COUNT_POS] = int(obs['step_count'])
        game_info[0, DONE_POS] = 0
        player1row = np.zeros((1, board_size)).astype(np.uint16)
        player1row[0, AMMO_POS] = int(obs['ammo'])
        player1row[0, BLAST_STRENGTH_POS] = int(obs['blast_strength'])
        player1row[0, CAN_KICK_POS] = int(obs['can_kick'])
        player1row[0, ALIVE_POS] = 1
        player1row[0, ROW_POS] = agent_0_pos[0]
        player1row[0, COL_POS] = agent_0_pos[1]
        player2row = np.zeros((1, board_size)).astype(np.uint16)
        player2row[0, AMMO_POS] = 1
        player2row[0, BLAST_STRENGTH_POS] = constants.DEFAULT_BLAST_STRENGTH
        player2row[0, CAN_KICK_POS] = False
        player2row[0, ALIVE_POS] = 1
        player2row[0, ROW_POS] = agent_1_pos[0]
        player2row[0, COL_POS] = agent_1_pos[1]
        bomb = np.zeros((1, board_size)).astype(np.uint16)
        game_data = np.vstack([game_data, game_info, player1row, player2row])

        return game_data

    @staticmethod
    def update(game_data, obs, my_id):
        enemy_id = 0
        if my_id == 0:
            enemy_id = 1

        step_count = EnvSimulator._get_game_value(game_data, STEP_COUNT_POS)

        if game_data.shape[1] != len(obs['board']):
            raise ValueError('Invalid update: boardsize different!')
        if step_count + 1 != int(obs['step_count']) and (step_count != 0 or int(obs['step_count']) != 0):
            raise ValueError('Invalid update: missed step count!')
        EnvSimulator._set_game_value(game_data, STEP_COUNT_POS, obs['step_count'])

        new_board = EnvSimulator._get_game_data_from_obs(obs)

        new_board = EnvSimulator.get_board(game_data.shape[1], obs['board'])
        new_bomb_life = EnvSimulator.get_board(game_data.shape[1], obs['bomb_life'], 0)

        # get actions
        actions = {}
        for agent_id in [10, 11]:
            old_pos = EnvSimulator.get_position(game_data, agent_id, True)
            new_pos = EnvSimulator.get_position(new_board, agent_id + 10, True)

            if old_pos != new_pos:
                actions[agent_id] = EnvSimulator.get_direction(old_pos, new_pos).value
            elif new_bomb_life[new_pos] == constants.DEFAULT_BOMB_LIFE:
                actions[agent_id] = constants.Action.Bomb.value
            else:
                actions[agent_id] = constants.Action.Stop.value

        EnvSimulator.act(game_data, actions)

        reset = False

        # compare boards
        if not EnvSimulator.boards_equal(EnvSimulator.get_game_data_board(game_data), new_board, True):
            a1bomb, a2bomb, kick, flame = EnvSimulator.get_boards_differences(
                EnvSimulator.get_game_data_board(game_data), new_board)
            #print(a1bomb, a2bomb, kick, flame)
            if a1bomb and my_id != 0:
                ammo = EnvSimulator._get_agent_value(game_data, 0, AMMO_POS)
                EnvSimulator._set_agent_value(game_data, 0, AMMO_POS, ammo+1)
            elif a2bomb and my_id != 1:
                ammo = EnvSimulator._get_agent_value(game_data, 1, AMMO_POS)
                EnvSimulator._set_agent_value(game_data, 1, AMMO_POS, ammo + 1)
            elif kick and EnvSimulator._get_agent_value(game_data, my_id, CAN_KICK_POS) == int(obs['can_kick']):
                EnvSimulator._set_agent_value(game_data, enemy_id, CAN_KICK_POS, 1)
            elif flame and EnvSimulator._get_agent_value(game_data, my_id, BLAST_STRENGTH_POS) == int(obs['blast_strength']):
                blast = EnvSimulator._get_agent_value(game_data, enemy_id, BLAST_STRENGTH_POS)
                EnvSimulator._set_agent_value(game_data, enemy_id, BLAST_STRENGTH_POS, blast+1)
            reset = True

        EnvSimulator._set_agent_value(game_data, enemy_id, AMMO_POS, int(obs['ammo']))
        EnvSimulator._set_agent_value(game_data, enemy_id, BLAST_STRENGTH_POS, int(obs['blast_strength']))
        EnvSimulator._set_agent_value(game_data, enemy_id, CAN_KICK_POS, int(obs['can_kick']))

        # update board because of items
        game_data[0:game_data.shape[1], 0:game_data.shape[1]] = new_board

        return game_data, actions, reset

    @staticmethod
    def _get_game_data_from_obs(obs):
        board_size = len(obs['board'])
        board = EnvSimulator.get_board(board_size, obs['board'])
        blast_strength = obs['bomb_blast_strength']
        bomb_life = obs['bomb_life']

        for row in range(len(board)):
            for col in range(len(board[0])):
                if (board[row, col] == 10 or board[row, col] == 11) and blast_strength[row, col] > 0.0:
                    # agent over bomb
                    value = 10000 + (board[row, col]-7)*1000 + int(blast_strength[row, col])*10 + int(bomb_life[row, col])
                    board[row, col] = value
                if board[row, col] == 3: # bomb
                    agent_id = 0
                    value = 10000 + (board[row, col]-7)*1000 + int(blast_strength[row, col])*10 + int(bomb_life[row, col])

        return

    @staticmethod
    def get_game_data_board(game_data):
        return game_data[0:game_data.shape[1], 0:game_data.shape[1]]

    @staticmethod
    def act(game_data, actions):
        MIN_FIRE = 20
        AGENT_0 = 10
        AGENT_1 = 11

        if EnvSimulator.get_done(game_data):
            return

        #print(game_data, actions)

        # move objects
        pos_agent0_prev = None
        pos_agent0 = None
        pos_agent1_prev = None
        pos_agent1 = None
        pos_bomb_prev = []
        for row in range(game_data.shape[1]):
            for col in range(game_data.shape[1]):
                if EnvSimulator._is_fire(game_data, (row, col)):
                    game_data[row, col] -= 1
                    if game_data[row, col] == MIN_FIRE:
                        game_data[row, col] = 0
                elif game_data[row, col] == AGENT_1 or game_data[row, col] >= 14000:
                    pos_agent1_prev = (row, col)
                    pos_agent1 = EnvSimulator.handle_agent_move(game_data, 1, row, col, actions[1])
                elif game_data[row, col] == AGENT_0 or game_data[row, col] >= 13000:
                    pos_agent0_prev = (row, col)
                    pos_agent0 = EnvSimulator.handle_agent_move(game_data, 0, row, col, actions[0])
                if game_data[row, col] >= 10000:
                    pos_bomb_prev.append((row, col))

        if pos_agent0 == pos_agent1:
            pos_agent0 = pos_agent0_prev
            pos_agent1 = pos_agent1_prev

        # move bombs
        pos_bomb = []
        change = False
        invalid_values = [constants.Item.Rigid.value, constants.Item.Wood.value, constants.Item.Kick,
                          constants.Item.IncrRange, constants.Item.ExtraBomb]
        for bomb_pos in pos_bomb_prev:
            bomb = game_data[bomb_pos]
            direction = int((bomb % 1000) / 100)
            if direction == 0 and bomb_pos == pos_agent0:
                if pos_agent0 != pos_agent0_prev:  # kick bomb
                    direction = EnvSimulator.get_direction(pos_agent0_prev, pos_agent0).value
                elif int((bomb % 10000) / 1000) != 1 and int((bomb % 10000) / 1000) != 3:
                    raise ValueError("Fatal Error")
            elif direction == 0 and bomb_pos == pos_agent1:
                if pos_agent1 != pos_agent1_prev:  # kick bomb
                    direction = EnvSimulator.get_direction(pos_agent1_prev, pos_agent1).value
                elif int((bomb % 10000) / 1000) != 2 and int((bomb % 10000) / 1000) != 4:
                    raise ValueError("Fatal Error")

            new_bomb_pos = bomb_pos
            if direction > 0:
                change = True
                row, col = bomb_pos
                if EnvSimulator._is_valid_direction(game_data, row, col, direction, invalid_values):
                    new_bomb_pos = utility.get_next_position(bomb_pos, constants.Action(direction))
                if (row, col) == pos_agent0 or (row, col) == pos_agent1:
                    new_bomb_pos = bomb_pos

            pos_bomb.append(new_bomb_pos)

        while change:
            change = False
            # bomb <-> bomb
            for i in range(len(pos_bomb)):
                pos = pos_bomb[i]
                for j in range(len(pos_bomb)):
                    if i != j and pos == pos_bomb[j]:
                        pos_bomb[i] = pos_bomb_prev[i]
                        pos_bomb[j] = pos_bomb_prev[j]
                        change = True
                if pos_bomb[i] == pos_agent0 and (pos_bomb[i] != pos_bomb_prev[i] or pos_agent0 != pos_agent0_prev):
                    pos_agent0 = pos_agent0_prev
                    pos_bomb[i] = pos_bomb_prev[i]
                    change = True
                elif pos_bomb[i] == pos_agent1 and (pos_bomb[i] != pos_bomb_prev[i] or pos_agent1 != pos_agent1_prev):
                    pos_agent1 = pos_agent1_prev
                    pos_bomb[i] = pos_bomb_prev[i]
                    change = True

        for i in range(len(pos_bomb)):
            cur_value = game_data[pos_bomb_prev[i]]
            life = int(cur_value % 10) - 1
            if 20 < game_data[pos_bomb[i]] < 30:
                life = 0
            strength = int((cur_value % 100) / 10)
            direction = EnvSimulator.get_direction(pos_bomb[i], pos_bomb_prev[i]).value
            player = int((cur_value % 10000) / 1000)
            if player > 2:
                player -= 2
            if pos_agent0 == pos_bomb[i] or pos_agent1 == pos_bomb[i]:
                player += 2

            game_data[pos_bomb_prev[i]] = 0
            game_data[pos_bomb[i]] = 10000 + player * 1000 + direction * 100 + strength * 10 + life

        # set agent
        #print(pos_agent0, pos_agent1)
        EnvSimulator._agent_collect(game_data, 0, pos_agent0)
        EnvSimulator._agent_collect(game_data, 1, pos_agent1)

        if pos_agent0_prev != pos_agent0:
            if game_data[pos_agent0_prev] < 10000:
                game_data[pos_agent0_prev] = 0
            if EnvSimulator._is_fire(game_data, pos_agent0):
                EnvSimulator._agent_died(game_data, 0)
            else:
                game_data[pos_agent0] = AGENT_0

        if pos_agent1_prev != pos_agent1:
            if game_data[pos_agent1_prev] < 10000:
                game_data[pos_agent1_prev] = 0
            if EnvSimulator._is_fire(game_data, pos_agent1):
                EnvSimulator._agent_died(game_data, 1)
            else:
                game_data[pos_agent1] = AGENT_1

        # fire bombs
        fire = True
        while fire:
            fire = False
            for bomb in pos_bomb:
                bomb_value = game_data[bomb]
                if int(bomb_value % 10) == 0:
                    strength = int((bomb_value % 100) / 10)
                    EnvSimulator._set_fire(game_data, bomb[0], bomb[1], True)
                    EnvSimulator._fire_bomb(game_data, bomb[0], bomb[1], 0, 1, strength - 1)  # right
                    EnvSimulator._fire_bomb(game_data, bomb[0], bomb[1], 0, -1, strength - 1)  # left
                    EnvSimulator._fire_bomb(game_data, bomb[0], bomb[1], 1, 0, strength - 1)  # down
                    EnvSimulator._fire_bomb(game_data, bomb[0], bomb[1], -1, 0, strength - 1)  # up
                    fire = True

        #print('result: ', game_data)

    @staticmethod
    def handle_agent_move(game_data, agent_id, row, col, action):
        if action == constants.Action.Stop.value:
            return row, col
        elif action == constants.Action.Bomb.value:
            ammo = EnvSimulator._get_agent_value(game_data, agent_id, AMMO_POS)
            if game_data[row, col] < 10000 and ammo > 0:
                game_data[row, col] = 10009 + (agent_id + 3) * 1000 + EnvSimulator._get_agent_value(game_data, agent_id, BLAST_STRENGTH_POS) * 10
                EnvSimulator._set_agent_value(game_data, agent_id, AMMO_POS, ammo-1)
            return row, col
        else:
            invalid_values = [constants.Item.Rigid.value, constants.Item.Wood.value]
            if EnvSimulator._is_valid_direction(game_data, row, col, action, invalid_values):
                return utility.get_next_position((row, col), constants.Action(action))
            else:
                return row, col

    @staticmethod
    def _agent_collect(game_data, agent_id, pos):
        item = game_data[pos]
        if item == constants.Item.Kick.value:
            EnvSimulator._set_agent_value(game_data, agent_id, CAN_KICK_POS, 1)
        elif item == constants.Item.ExtraBomb.value:
            cur_ammo = EnvSimulator._get_agent_value(game_data, agent_id, AMMO_POS)
            EnvSimulator._set_agent_value(game_data, agent_id, AMMO_POS, cur_ammo + 1)
        elif item == constants.Item.IncrRange.value:
            cur_range = EnvSimulator._get_agent_value(game_data, agent_id, BLAST_STRENGTH_POS)
            EnvSimulator._set_agent_value(game_data, agent_id, BLAST_STRENGTH_POS, cur_range + 1)

    @staticmethod
    def _position_on_board(game_data, row, col):
        return all([game_data.shape[1] > row, game_data.shape[1] > col, row >= 0, col >= 0])

    @staticmethod
    def _is_fire(game_data, pos):
        return 20 < game_data[pos] < 30

    @staticmethod
    def _fire_bomb(game_data, row, col, row_off, col_off, strength):
        if strength <= 0:
            return
        next_row = row + row_off
        next_col = col + col_off
        if not EnvSimulator._position_on_board(game_data, next_row, next_col):
            return
        if utility.position_in_items(game_data, (next_row, next_col), [constants.Item.Rigid, constants.Item.Wood]):
            return

        EnvSimulator._set_fire(game_data, next_row, next_col, False)

        EnvSimulator._fire_bomb(game_data, next_row, next_col, row_off, col_off, strength - 1)

    @staticmethod
    def _set_fire(game_data, row, col, first):
        prev_value = game_data[row, col]
        if prev_value > 14000 or prev_value == 11:
            EnvSimulator._agent_died(game_data, 1)
        if prev_value > 13000 or prev_value == 10:
            EnvSimulator._agent_died(game_data, 0)
        if not first and prev_value > 10000:
            prev_value -= int(prev_value % 10)
        else:
            if first and prev_value > 10000:
                # increase ammo
                player = int((prev_value % 10000) / 1000)
                if player == 1 or player == 3:
                    player = 0
                else:
                    player = 1
                ammo = EnvSimulator._get_agent_value(game_data, player, AMMO_POS)
                EnvSimulator._set_agent_value(game_data, player, AMMO_POS, ammo+1)
            game_data[row, col] = 22

    @staticmethod
    def _agent_died(game_data, agent_id):
        EnvSimulator._set_agent_value(game_data, agent_id, ALIVE_POS, 0)
        EnvSimulator._set_game_value(game_data, DONE_POS, 1)

    @staticmethod
    def _is_valid_direction(board, row, col, direction, invalid_values=None):
        if invalid_values is None:
            invalid_values = [item.value for item in [constants.Item.Rigid, constants.Item.Wood]]

        if constants.Action(direction) == constants.Action.Stop:
            return True

        if constants.Action(direction) == constants.Action.Up:
            return row - 1 >= 0 and board[row - 1][col] not in invalid_values

        if constants.Action(direction) == constants.Action.Down:
            return row + 1 < len(board) and board[row + 1][col] not in invalid_values

        if constants.Action(direction) == constants.Action.Left:
            return col - 1 >= 0 and board[row][col - 1] not in invalid_values

        if constants.Action(direction) == constants.Action.Right:
            return col + 1 < len(board[0]) and board[row][col + 1] not in invalid_values

        raise constants.InvalidAction("We did not receive a valid direction: ", direction)

    @staticmethod
    def get_direction(position, next_position):
        if position == next_position:
            return constants.Action.Stop

        x, y = position
        next_x, next_y = next_position
        if x == next_x:
            if y < next_y:
                return constants.Action.Right
            else:
                return constants.Action.Left
        elif y == next_y:
            if x < next_x:
                return constants.Action.Down
            else:
                return constants.Action.Up
        raise constants.InvalidAction(
            "We did not receive a valid position transition.")

    @staticmethod
    def _get_agent_value(game_data, agent_id, value):
        return game_data[game_data.shape[0] - 2 + agent_id, value]

    @staticmethod
    def _set_agent_value(game_data, agent_id, value, val):
        game_data[game_data.shape[0] - 2 + agent_id, value] = val

    @staticmethod
    def _get_game_value(game_data, value):
        return game_data[game_data.shape[0] - 3, value]

    @staticmethod
    def _set_game_value(game_data, value, val):
        game_data[game_data.shape[0] - 3, value] = val

    @staticmethod
    def get_done(game_data):
        return bool(EnvSimulator._get_game_value(game_data, DONE_POS))

    @staticmethod
    def get_alive(game_data):
        alive = {0: bool(game_data[game_data.shape[0] - 2, ALIVE_POS]),
                 1: bool(game_data[game_data.shape[0] - 1, ALIVE_POS])}
        return alive

    @staticmethod
    def get_board(board_size, board_array, init_value=constants.Item.Passage.value):
        board = np.ones((board_size, board_size)).astype(np.uint16)
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
    def get_valid_actions(board, flames, bombs, agent, actions):
        return actions

    @staticmethod
    def boards_equal(board1, board2, ignore_items):
        if ignore_items:
            board1 = copy.deepcopy(board1)
            board2 = copy.deepcopy(board2)
            board1[board1 == constants.Item.ExtraBomb.value] = constants.Item.Passage.value
            board1[board1 == constants.Item.IncrRange.value] = constants.Item.Passage.value
            board1[board1 == constants.Item.Kick.value] = constants.Item.Passage.value
            board2[board2 == constants.Item.ExtraBomb.value] = constants.Item.Passage.value
            board2[board2 == constants.Item.IncrRange.value] = constants.Item.Passage.value
            board2[board2 == constants.Item.Kick.value] = constants.Item.Passage.value

        comparison = (board1 == board2)
        return comparison.all()

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
                if prev_item is constants.Item.Agent1 and new_item is constants.Item.Bomb:
                    a1bomb = True
                elif prev_item is constants.Item.Agent2 and new_item is constants.Item.Bomb:
                    a2bomb = True
                elif prev_item is constants.Item.Passage and new_item is constants.Item.Bomb:
                    kick = True
                elif new_item is constants.Item.Flames:
                    flame = True
                else:
                    raise ValueError('Invalid difference between maps.')
        else:
            print(comparison, "diffs: ", diffs)

        return a1bomb, a2bomb, kick, flame

    @staticmethod
    def get_game_state(game_data):
        return game_data, EnvSimulator.get_done(game_data)

    @staticmethod
    def get_game_data(game_state):
        return copy.deepcopy(game_state)
