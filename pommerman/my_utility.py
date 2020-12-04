import numpy as np
from pommerman import forward_model
from pommerman import constants
from pommerman import characters
from pommerman import graphics
from pommerman.agents import env_simulator

from pyglet.image.codecs.png import PNGImageEncoder
import pyglet
import matplotlib.pyplot as plt

def get_gamestate(board, bomb_info):
    board = np.array(board)

    pos1 = get_position(board, 10, True)
    pos2 = get_position(board, 11, True)
    b_pos = get_position(board, constants.Item.Bomb.value, False)

    if len(b_pos) != len(bomb_info):
        ValueError('Invalid bomb_info length!')

    bombs = []
    for b in range(len(bomb_info)):
        inf = {
            "position": b_pos[b],
            "bomber": bomb_info[b][0],
            "life": bomb_info[b][1],
            "blast_strength": bomb_info[b][2],
            "moving_direction": bomb_info[b][3]
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
        "step_count": 1
    }
    return gamestate

def get_gamedata(gamestate, game_type):
    game_data = env_simulator.GameData()
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
        game_data.bombs.append(bomb)

    # flames
    game_data.flames = []
    for f in gamestate['flames']:
        flame = characters.Flame(**f)
        game_data.flames.append(flame)

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

def get_texture(file, game_data, size):
    rm = graphics.ResourceManager(game_data.game_type)
    h = game_data.board.shape[0]
    w = game_data.board.shape[1]

    if size:
        tile_width = int(size[0] / w)
        tile_height = int(size[1] / h)
    else:
        ex = rm.tile_from_state_value(0)
        tile_width = ex.width
        tile_height = ex.height

    texture = pyglet.image.Texture.create(width=tile_width * w, height=tile_height * h)

    board = game_data.board
    for row in range(h):
        for col in range(w):
            x = col * tile_width
            y = ((h-1) * tile_height) - row * tile_height
            tile_state = board[row][col]
            if tile_state == constants.Item.Bomb.value:
                bomb_life = get_bomb_life(game_data, row, col)
                tile = rm.get_bomb_tile(bomb_life)
            else:
                tile = rm.tile_from_state_value(tile_state)
            img = scale_img(tile, tile_width, tile_height)

            texture.blit_into(img.get_image_data(), x=x, y=y, z=0)

    texture.save(file=file, encoder=PNGImageEncoder()) # filename='C:\\tmp\\tmp.png')
    file.seek(0)


def get_bomb_life(game_data, row, col):
    for bomb in game_data.bombs:
        x, y = bomb.position
        if x == row and y == col:
            return bomb.life

def scale_img(image, width, height):
    #sprite = pyglet.sprite.Sprite(image)
    #sprite.update(scale=10)
    #image = sprite.image
    return image