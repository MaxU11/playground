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
import atexit
from datetime import datetime
import os
import random
import sys
import copy
import time
import csv

import argparse
import numpy as np

from pommerman import helpers
from pommerman import make
from pommerman import utility
from pommerman import constants

from pommerman.agents import SimpleAgent
from pommerman.agents import UcbMCTSAgent
from pommerman.agents import UcbLimitMCTSAgent
from pommerman.agents import UcbMRMCTSAgent
from pommerman.agents import UcbMRLimitMCTSAgent
from pommerman.agents.abstract_mcts_skeleton import AbstractMCTSSkeleton

def run_tournament(tournament_name, agent_pool1, agent_pool2, match_count, seed=None):
    '''Wrapper to help start the game'''
    config = 'OneVsOne-v0'
    record_pngs_dir = f'C:/tmp/Results/PNGS'
    record_json_dir = f'C:/tmp/Results/JSON'
    csv_dir = f'C:/tmp/Results/CSV'
    game_state_file = None
    render_mode = 'human'
    do_sleep = False
    render = False

    game_details = [['p1','p2','result','winner','time','steps', 'add_info_p1', 'add_info_p2']]

    duel_num = 0
    total_duels = len(agent_pool1) * len(agent_pool2)
    game_num = 0
    total_games = total_duels * match_count * 2
    for p1_a in agent_pool1:
        for p2_a in agent_pool2:
            duel_num += 1
            print(f'Duel {duel_num}/{total_duels}')

            for d in range(2):
                if d == 0:
                    agents = [p1_a[1](**p1_a[2]), p2_a[1](**p2_a[2])]
                    agent_names = [p1_a[0], p2_a[0]]
                else:
                    agents = [p2_a[1](**p2_a[2]), p1_a[1](**p1_a[2])]
                    agent_names = [p2_a[0], p1_a[0]]

                env = make(config, agents, game_state_file, render_mode=render_mode)
                if seed is None:
                    # Pick a random seed between 0 and 2^31 - 1
                    seed = random.randint(0, np.iinfo(np.int32).max)
                np.random.seed(seed)
                random.seed(seed)
                env.seed(seed)

                for i in range(match_count):
                    game_num += 1

                    if record_pngs_dir:
                        record_pngs_dir_ = f'{record_pngs_dir}/{tournament_name}/{agent_names[0]}_vs_{agent_names[1]}_{i+1}'
                    if record_json_dir:
                        record_json_dir_ = f'{record_json_dir}/{tournament_name}/{agent_names[0]}_vs_{agent_names[1]}_{i+1}'

                    start = time.time()
                    info, steps = run(env, agent_names, config, render, do_sleep, record_pngs_dir_, record_json_dir_)
                    total_time = time.time() - start
                    winner = -1
                    if info['result'] == constants.Result.Win:
                        winner = int(info['winners'][0])

                    agent_info_1 = {}
                    agent_info_2 = {}
                    if isinstance(agents[0], AbstractMCTSSkeleton):
                        agents[0].get_agent_info(agent_info_1)
                    if isinstance(agents[1], AbstractMCTSSkeleton):
                        agents[1].get_agent_info(agent_info_2)

                    game_details.append([agent_names[0], agent_names[1], info['result'], winner, total_time, steps, agent_info_1, agent_info_2])

                    print(f"-- {game_num} / {total_games} Result: ", game_details[-1])

                atexit.register(env.close)

    if not os.path.isdir(csv_dir):
        os.makedirs(csv_dir)
    f = open(f'{csv_dir}/{tournament_name}.csv', 'w')

    with f:
        writer = csv.writer(f, delimiter=';')
        for row in game_details:
            writer.writerow(row)


def run(env, agent_names, config, render, do_sleep, record_pngs_dir=None, record_json_dir=None):
    '''Runs a game'''
    if record_pngs_dir and not os.path.isdir(record_pngs_dir):
        os.makedirs(record_pngs_dir)
    if record_json_dir and not os.path.isdir(record_json_dir):
        os.makedirs(record_json_dir)

    obs = env.reset()
    done = False

    steps = 0
    while not done:
        if render:
            env.render(
                record_pngs_dir=record_pngs_dir,
                record_json_dir=record_json_dir,
                do_sleep=do_sleep)
        if render is False and record_json_dir:
            env.save_json(record_json_dir)
            time.sleep(1.0 / env._render_fps)
        actions = env.act(obs)
        steps += 1
        obs, reward, done, info = env.step(actions)

    if render:
        env.render(
            record_pngs_dir=record_pngs_dir,
            record_json_dir=record_json_dir,
            do_sleep=do_sleep)
        if do_sleep:
            time.sleep(5)
        env.render(close=True)

    if render is False and record_json_dir:
        env.save_json(record_json_dir)
        time.sleep(1.0 / env._render_fps)

    if record_json_dir:
        finished_at = datetime.now().isoformat()
        utility.join_json_state(record_json_dir, agent_names, finished_at,
                                config, info)

    return info, steps


def run_simple_vs_ucb():
    agent_pool1 = []
    agent_pool2 = []

    # create agents
    agent_simple = SimpleAgent
    agent_pool1.append(('SimpleAgent', agent_simple, {}))

    for i in range(100):
        agent_ucb = UcbMCTSAgent
        kwargs = {'expandTreeRollout': False,
                  'maxIterations': i + 1,
                  'maxTime': 0.0,
                  'discountFactor': 0.9999,
                  'depthLimit': None,
                  'C': 0.5}
        agent_pool2.append((
                           f'AgentUCB_iter{kwargs["maxIterations"]}_df{kwargs["discountFactor"]}_dl{kwargs["depthLimit"]}_ex{kwargs["expandTreeRollout"]}_c{kwargs["C"]}',
                           agent_ucb, kwargs))

    # Tournament Settings
    tournament_name = 'Simple_Against_UCB_' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    match_count = 10

    run_tournament(tournament_name, agent_pool1, agent_pool2, match_count)


def run_ucb_depth_test():
    agent_pool1 = []
    agent_pool2 = []

    # create agents
    agent_ucb = UcbMCTSAgent
    kwargs = {'expandTreeRollout': False,
              'maxIterations': 20,
              'maxTime': 0.0,
              'discountFactor': 0.9999,
              'depthLimit': None,
              'C': 0.5}
    agent_pool1.append((f'AgentUCB_iter{kwargs["maxIterations"]}_df{kwargs["discountFactor"]}_dl{kwargs["depthLimit"]}_ex{kwargs["expandTreeRollout"]}_c{kwargs["C"]}',
                           agent_ucb, kwargs))

    for i in range(50):
        agent_ucb = UcbMCTSAgent
        kwargs = {'expandTreeRollout': False,
                  'maxIterations': 20,
                  'maxTime': 0.0,
                  'discountFactor': 0.9999,
                  'depthLimit': (50 - i),
                  'C': 0.5}
        agent_pool2.append((
                           f'AgentUCB_iter{kwargs["maxIterations"]}_df{kwargs["discountFactor"]}_dl{kwargs["depthLimit"]}_ex{kwargs["expandTreeRollout"]}_c{kwargs["C"]}',
                           agent_ucb, kwargs))

    # Tournament Settings
    tournament_name = 'UCB_Depth_Limit_Test' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    match_count = 10

    run_tournament(tournament_name, agent_pool1, agent_pool2, match_count)


def main():
    run_simple_vs_ucb()


if __name__ == "__main__":
    main()