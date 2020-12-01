import numpy as np
from pommerman import forward_model
from pommerman import constants
from pommerman import characters
from pommerman import graphics

def get_gamestate(board, bomb_info):
    board = np.array(board)

    pos1 = get_position(board, 10, True)
    pos2 = get_position(board, 11, True)
    b_pos = get_position(board, constants.Item.Bomb.value, False)

    if len(b_pos) != len(bomb_info):
        ValueError('Invalid bomb_info length!')

    bombs = []
    for b in range(list(bomb_info)):
        inf = {
            "position": b_pos[b],
            "bomber_id": bomb_info[0],
            "life": bomb_info[1],
            "blast_strength": bomb_info[2],
            "moving_direction": bomb_info[3]
        }
        bombs.append(inf)

    gamestate = {
        "agents": [
            {
                "agent_id": 0,
                "is_alive": True,
                "position": pos1,
                "ammo": 1,
                "blast_strength": 2,
                "can_kick": False
            },
            {
                "agent_id": 1,
                "is_alive": True,
                "position": pos2,
                "ammo": 1,
                "blast_strength": 2,
                "can_kick": False
            }],
        "board": board,
        "board_size": len(board),
        "bombs": bombs,
        "flames": [],
        "intended_actions": [0, 0],
        "items": [],
        "step_count": "1"
    }
    return gamestate

def get_gamedata(gamestate, game_type):
    game_data = object
    game_data.board_size = gamestate['board_size']
    game_data.step_count = gamestate['step_count'] - 1
    game_data.max_steps = 800
    game_data.game_type = game_type
    game_data.simulation_bomb_life = None

    # board
    game_data.board = gamestate['board']
    # items
    game_data.items = {}
    # agents
    game_data.agents = []
    for a in gamestate['agents']:
        id = a['agent_id']
        board_id = id + 10
        agent = characters.Bomber(id, game_data.game_type)
        agent.set_start_position(get_position(game_data.board, board_id, True))
        agent.reset(a['ammo'], a['is_alive'], a['blast_strength'], a['can_kick'])
        game_data.agents.append(agent)

    # bombs
    game_data.bombs = []
    for b in gamestate['bombs']:
        bomb = characters.Bomb(**b)

    # flames
    game_data.flames = []
    for f in gamestate['flames']:
        flame = characters.Flame(**f)

    # done
    game_data.done = forward_model.ForwardModel.get_done(game_data.agents, game_data.step_count,
                                                         game_data.max_steps, game_data.game_type, None)

    return game_data


def get_board(board_size, board_array, init_value=constants.Item.Passage.value):
    board = np.ones((board_size, board_size)).astype(np.uint8)
    board *= init_value
    for x in range(board_size):
        for y in range(board_size):
            board[x, y] = board_array[x][y]
    return board


def get_position(board, item, is_single_pos):
    pos = np.where(board == item)
    pos = list(zip(pos[0], pos[1]))
    if is_single_pos:
        if len(pos) != 1:
            raise ValueError("Invalid pos count!")
        return pos[0]
    else:
        return pos

def render_image(self, game_data, mode=None):
    mode = mode or self._mode or 'human'

    if mode == 'rgb_array':
        rgb_array = graphics.PixelViewer.rgb_array(
            game_data.board, game_data.board_size, game_data.agents,
            False, game_data.board_size)
        return rgb_array[0]

    if mode == 'rgb_pixel':
        _viewer = graphics.PixelViewer(
            board_size=game_data.board_size,
            agents=game_data.agents,
            agent_view_size=game_data.board_size,
            partially_observable=False)
    else:
        _viewer = graphics.PommeViewer(
            board_size=game_data._oard_size,
            agents=game_data.agents,
            partially_observable=False,
            agent_view_size=game_data.board_size,
            game_type=game_data.game_type)

        self._viewer.set_board(game_data.board)
        self._viewer.set_agents(game_data.agents)
        self._viewer.set_step(game_data.step_count)
        self._viewer.set_bombs(game_data.bombs)
        self._viewer.render()

        # Register all agents which need human input with Pyglet.
        # This needs to be done here as the first `imshow` creates the
        # window. Using `push_handlers` allows for easily creating agents
        # that use other Pyglet inputs such as joystick, for example.
        for agent in self._agents:
            if agent.has_user_input():
                self._viewer.window.push_handlers(agent)

    return _viewer.get_image()