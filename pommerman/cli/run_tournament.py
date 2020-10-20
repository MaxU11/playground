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

from .Tournament import run_tournament

from pommerman.agents import SimpleAgent
from pommerman.agents import UcbMCTSAgent
from pommerman.agents import UcbLimitMCTSAgent
from pommerman.agents import UcbMRMCTSAgent
from pommerman.agents import UcbMRLimitMCTSAgent


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


def main():
    run_ucb_rnd_vs_mcts()


if __name__ == "__main__":
    main()