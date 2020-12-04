"""Run a battle among agents.

Call this with a config, a game, and a list of agents. The script will start separate threads to operate the agents
and then report back the result.

An example with all four test agents running ffa:
python run_battle.py --agents=test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent --config=PommeFFACompetition-v0

An example with one player, two random agents, and one test agent:
python run_battle.py --agents=player::arrows,test::agents.SimpleAgent,random::null,random::null --config=PommeFFACompetition-v0

An example with a docker agent:
python run_battle.py --agents=player::arrows,docker::pommerman/test-agent,random::null,random::null --config=PommeFFACompetition-v0
"""
from datetime import datetime

from pommerman.cli.Tournament import run_tournament
from pommerman.cli.Tournament import run_single_match

from pommerman.agents import SimpleAgent
from pommerman.agents import UcbMCTSAgent
from pommerman.agents import UcbLimitMCTSAgent
from pommerman.agents import UcbMRMCTSAgent
from pommerman.agents import UcbMRLimitMCTSAgent
from pommerman.agents import NN_Agent

from pommerman.NN.pommerman_neural_net import PommermanNNet
from pommerman import constants


def run_simple_vs_ucb():
    agent_pool1 = []
    agent_pool2 = []

    # create agents
    agent_simple = SimpleAgent
    agent_pool1.append(('SimpleAgent', agent_simple, {}))

    iters = [1] + [(i+1) * 20 for i in range(5)]
    for i in iters:
        agent_ucb = UcbLimitMCTSAgent
        kwargs = {'expandTreeRollout': False,
                  'maxIterations': i,
                  'maxTime': 0.0,
                  'discountFactor': 0.9999,
                  'depthLimit': None,
                  'C': 0.5}
        agent_pool2.append((
                           f'AgentUCBLimit_iter{kwargs["maxIterations"]}_df{kwargs["discountFactor"]}_dl{kwargs["depthLimit"]}_ex{kwargs["expandTreeRollout"]}_c{kwargs["C"]}',
                           agent_ucb, kwargs))

    # Tournament Settings
    tournament_name = 'Simple_Against_UCBLimit_' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    match_count = 50

    run_tournament(tournament_name, agent_pool1, agent_pool2, match_count)


def run_ucb_rnd_vs_limit():
    agent_pool1 = []
    agent_pool2 = []

    # create agents
    iters = [1] + [(i + 1) * 20 for i in range(5)]
    for i in iters:
        agent_ucb = UcbMCTSAgent
        kwargs = {'expandTreeRollout': False,
                  'maxIterations': i,
                  'maxTime': 0.0,
                  'discountFactor': 0.9999,
                  'depthLimit': None,
                  'C': 0.5}
        agent_pool1.append((
            f'AgentUCB_iter{kwargs["maxIterations"]}_df{kwargs["discountFactor"]}_dl{kwargs["depthLimit"]}_ex{kwargs["expandTreeRollout"]}_c{kwargs["C"]}',
            agent_ucb, kwargs))

    for i in iters:
        agent_ucb = UcbLimitMCTSAgent
        kwargs = {'expandTreeRollout': False,
                  'maxIterations': i,
                  'maxTime': 0.0,
                  'discountFactor': 0.9999,
                  'depthLimit': None,
                  'C': 0.5}
        agent_pool2.append((
            f'AgentUCBLimit_iter{kwargs["maxIterations"]}_df{kwargs["discountFactor"]}_dl{kwargs["depthLimit"]}_ex{kwargs["expandTreeRollout"]}_c{kwargs["C"]}',
            agent_ucb, kwargs))

    # Tournament Settings
    tournament_name = 'UCB_Depth_Limit_Test' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    match_count = 10

    run_tournament(tournament_name, agent_pool1, agent_pool2, match_count, False)


def run_ucb_rnd_vs_mcts():
    agent_pool1 = []
    agent_pool2 = []

    # create agents
    iters = [1] + [(i + 1) * 20 for i in range(5)]
    for i in iters:
        agent_ucb = UcbMCTSAgent
        kwargs = {'expandTreeRollout': False,
                  'maxIterations': i,
                  'maxTime': 0.0,
                  'discountFactor': 0.9999,
                  'depthLimit': None,
                  'C': 0.5}
        agent_pool1.append((
            f'AgentUCB_iter{kwargs["maxIterations"]}_df{kwargs["discountFactor"]}_dl{kwargs["depthLimit"]}_ex{kwargs["expandTreeRollout"]}_c{kwargs["C"]}',
            agent_ucb, kwargs))

    for i in iters:
        agent_ucb = UcbMRMCTSAgent
        kwargs = {'expandTreeRollout': False,
                  'maxIterations': 6,
                  'maxTime': 0.0,
                  'discountFactor': 0.9999,
                  'depthLimit': 26,
                  'C': 0.5,
                  'MRDepthLimit': 2}
        agent_pool2.append((
            f'AgentUCBMRMCTS_iter{kwargs["maxIterations"]}_df{kwargs["discountFactor"]}_dl{kwargs["depthLimit"]}_ex{kwargs["expandTreeRollout"]}_c{kwargs["C"]}',
            agent_ucb, kwargs))

    # Tournament Settings
    tournament_name = 'UCB_MR_VS_RND_Test' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    match_count = 10

    run_tournament(tournament_name, agent_pool1, agent_pool2, match_count, False)

def run_and_render_match():
    nn_args = {
        'input_channels': 8,
        'board_x': constants.BOARD_SIZE_ONE_VS_ONE,
        'board_y': constants.BOARD_SIZE_ONE_VS_ONE,

        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 10,  # 10,
        'batch_size': 64,
        'cuda': False,
        'num_channels': 512
    }
    agent_args = {
        'expandTreeRollout': False,
        'maxIterations': 100,
        'maxTime': 0.1,
        'discountFactor': 0.9999,
        'depthLimit': 26,
        'C': 1.0,
        'tempThreshold': 5
    }
    agent1 = NN_Agent(PommermanNNet(**nn_args), **agent_args)
    agent2 = NN_Agent(PommermanNNet(**nn_args), **agent_args)
    run_single_match(agent1, agent2, True)

import matplotlib.image as mpimg
from io import BytesIO
from io import StringIO
from pommerman import my_utility
def main():
    # Passage = 0, Rigid = 1, Wood = 2, Bomb = 3, Flames = 4
    # ExtraBomb = 6, IncrRange = 7, Kick = 8, Agent0 = 10, Agent1 = 11
    game_type = constants.GameType(4)

    board = [[0, 0, 2, 1, 1, 1],
             [0, 0, 10, 0, 0, 0],
             [2, 0, 0, 1, 3, 1],
             [1, 0, 1, 0, 0, 1],
             [1, 0, 0, 3, 0, 1],
             [1, 11, 1, 1, 1, 0]]
    bomb_info = [(0, 5, 2, None), (1, 3, 2, None)]

    game_state = my_utility.get_gamestate(board, bomb_info)
    game_data = my_utility.get_gamedata(game_state, game_type)

    data = BytesIO()
    my_utility.get_texture(data, game_data, None)



if __name__ == "__main__":
    main()