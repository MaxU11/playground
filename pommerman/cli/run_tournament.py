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
from pommerman.agents import SimpleCarefulAgent
from pommerman.agents import UcbMCTSAgent
from pommerman.agents import UcbLimitMCTSAgent
from pommerman.agents import UcbMRMCTSAgent
from pommerman.agents import UcbMRLimitMCTSAgent
from pommerman.agents import NN_Agent

from pommerman.NN.pommerman_neural_net import PommermanNNet
from pommerman import constants

def run_ucb_vs_ucb():
    agent_pool1 = []
    agent_pool2 = []

    # create agents
    for i in [30, 20, 10, 5]:
        agent_ucb1 = UcbMCTSAgent
        kwargs = {'maxIterations': 100, 'maxTime': 0.2, 'depthLimit': 0}
        agent_pool1.append((f'UcbMCTSAgent_noDL', agent_ucb1, kwargs))

        agent_ucb2 = UcbMCTSAgent
        kwargs = {'maxIterations': 100, 'maxTime': 0.2, 'depthLimit': i}
        agent_pool2.append((f'UcbMCTSAgent_DL{i}', agent_ucb2, kwargs))

    # Tournament Settings
    tournament_name = 'UCB_DepthLimit_Test' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    match_count = 20

    run_tournament(tournament_name, agent_pool1, agent_pool2, match_count, AllVsAll=False)

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

def compare_nnet_agents():
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
        'maxIterations': 25,
        'maxTime': 0.0,
        'discountFactor': 0.9999,
        'depthLimit': 26,
        'C': 1.0,
        'tempThreshold': 0
    }
    nnet1 = PommermanNNet(**nn_args)
    nnet1.load_checkpoint('C:/tmp/Model/analyse_1', 'nnet_6.pth.tar')

    nnet2 = PommermanNNet(**nn_args)

    agent_pool1 = [(f'Trained_NN_Agent', NN_Agent, {'nnet': nnet1, **agent_args})]
    agent_pool2 = [(f'Rnd_NN_Agent', NN_Agent, {'nnet': nnet2, **agent_args})]
    match_count = 40
    wins, ties, loss = run_tournament('tournament_name', agent_pool1, agent_pool2, int(match_count / 2), False, False, False)
    print('Ended: ', wins, ties, loss)

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
        'maxIterations': 30,
        'maxTime': 0.0,
        'discountFactor': 0.9999,
        'depthLimit': 30,
        'C': 0,
        'tempThreshold': 0,
        'useDomainKnowledge': False
    }
    nnet1 = PommermanNNet(**nn_args)
    nnet1.load_checkpoint('C:/tmp/Model/analyse_2', 'nnet_3.pth.tar')

    #nnet2 = PommermanNNet(**nn_args)
    #nnet2.load_checkpoint('C:/tmp/Model/analyse_1', 'nnet_6.pth.tar')

    #agent1 = NN_Agent(SimpleAgent(), **agent_args)
    #agent2 = NN_Agent(SimpleAgent(), **agent_args)
    agent1 = SimpleCarefulAgent()
    agent2 = NN_Agent(nnet1, **agent_args)
    run_single_match(agent1, agent2, True)

def run_simple_vs_simple2():
    agent_pool1 = []
    agent_pool2 = []

    # create agents
    agent_simple = SimpleAgent2
    agent_pool1.append(('SimpleAgent', agent_simple, {}))

    agent_simple2 = SimpleAgent2
    agent_pool2.append(('SimpleAgent2', agent_simple2, {}))

    # Tournament Settings
    tournament_name = 'Simple_Against_SimpleAgent2_' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    match_count = 1000

    run_tournament(tournament_name, agent_pool1, agent_pool2, match_count, only_one_side=True)

from pommerman import my_utility
from pommerman import forward_model
from gym import spaces
def testSimpleAgent():
    game_type = constants.GameType(4)

    board = [[0, 0, 2, 1, 1, 1],
             [0, 0, 0, 0, 0, 0],
             [2, 8, 0, 1, 0, 1],
             [1, 0, 1, 0, 10, 1],
             [1, 0, 3, 0,  0, 1],
             [1, 11, 1, 1, 1, 0]]
    bomb_info = [(0, 1, 2, None)]

    game_state = my_utility.get_gamestate(board, bomb_info)
    game_data = my_utility.get_gamedata(game_state, game_type)

    fm = forward_model.ForwardModel()

    obs = fm.get_observations(game_data.board, game_data.agents, game_data.bombs, game_data.flames,
                              False, None, game_data.game_type, None)

    simpel_agent = SimpleAgent()

    print(simpel_agent.act(obs[1], spaces.Discrete(6)))

def run_ucb_vs_simplecareful():
    agent_pool1 = []
    agent_pool2 = []

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
        'maxIterations': 80,
        'maxTime': 0.0,
        'discountFactor': 0.9999,
        'depthLimit': 0,
        'C': 0.1,
        'tempThreshold': 0,
        'useDomainKnowledge': False
    }
    nnet1 = PommermanNNet(**nn_args)
    nnet1.load_checkpoint('C:/tmp/Model/analyse_2', 'nnet_3.pth.tar')

    # create agents
    for i in [100, 120, 140]:
        agent_ucb1 = NN_Agent
        kwargs = agent_args.copy()
        kwargs['nnet'] = nnet1
        kwargs['maxIterations'] = i
        agent_pool1.append((f'AlphaZeroAgent_noDL_iter{i}', agent_ucb1, kwargs))

    agent_simple2 = SimpleAgent
    agent_pool2.append(('SimpleAgent', agent_simple2, {}))
    agent_pool1 = [('SimpleAgent', agent_simple2, {})]

    # Tournament Settings
    tournament_name = 'SimpleAgent_SimpleAgent_' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    match_count = 100

    run_tournament(tournament_name, agent_pool2, agent_pool1, match_count, only_one_side=False)

def big_tournament():
    agent_pool = []
    agent_pool1 = []
    agent_pool2 = []

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
        'maxIterations': 80,
        'maxTime': 0.0,
        'discountFactor': 0.9999,
        'C': 0.1,
        'tempThreshold': 0,
        'useDomainKnowledge': False
    }
    nnet_NORS = PommermanNNet(**nn_args)
    nnet_NORS.load_checkpoint('C:/tmp/Model/analyse_2', 'nnet_3.pth.tar')

    nnet_RS = PommermanNNet(**nn_args)
    nnet_RS.load_checkpoint('C:/tmp/Model/analyse_Reward', 'nnet_2.pth.tar')

    kwargs = agent_args.copy()
    kwargs['maxIterations'] = 60
    kwargs['depthLimit'] = 26
    agent_pool.append(('UCT-Agent', UcbMCTSAgent, kwargs))

    kwargs = agent_args.copy()
    kwargs['maxIterations'] = 40
    kwargs['depthLimit'] = 26
    agent_pool.append(('UCT-ActionPruning-Agent', UcbLimitMCTSAgent, kwargs))

    kwargs = agent_args.copy()
    kwargs['maxIterations'] = 5
    kwargs['depthLimit'] = 26
    agent_pool.append(('UCT-MiniMax-Agent', UcbMRLimitMCTSAgent, kwargs))

    kwargs = agent_args.copy()
    kwargs['nnet'] = nnet_NORS
    kwargs['maxIterations'] = 40
    agent_pool.append(('AlphaZero-Agent', NN_Agent, kwargs))

    kwargs = agent_args.copy()
    kwargs['nnet'] = nnet_RS
    kwargs['maxIterations'] = 40
    agent_pool.append(('AlphaZero-RewardShaping-Agent', NN_Agent, kwargs))

    start_i = 0
    game_num = 0
    for i, a in enumerate(agent_pool):
        for j in range(len(agent_pool)-i):
            game_num += 1
            if start_i <= game_num:
                print(i, j)
                agent_pool1.append(agent_pool[i])
                agent_pool2.append(agent_pool[i+j])

                print(agent_pool1[-1][0], 'vs', agent_pool2[-1][0])

    # Tournament Settings
    tournament_name = 'BigTournament' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    match_count = 25

    run_tournament(tournament_name, agent_pool2, agent_pool1, match_count, only_one_side=False, AllVsAll=False)

if __name__ == "__main__":
    #run_and_render_match()
    #run_simple_vs_simple2()
    #testSimpleAgent()
    run_ucb_vs_simplecareful()
    #big_tournament()