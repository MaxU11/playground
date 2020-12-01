import csv
import os
import random
import time
import atexit
import numpy as np
from datetime import datetime

from pommerman.agents.abstract_mcts_skeleton import AbstractMCTSSkeleton
from pommerman import utility
from pommerman import constants
from pommerman import make


def run(env, agent_names, config, render, do_sleep, record_pngs_dir=None, record_json_dir=None):
    '''Runs a game'''
    if record_pngs_dir and not os.path.isdir(record_pngs_dir):
        os.makedirs(record_pngs_dir)
    if record_json_dir and not os.path.isdir(record_json_dir):
        os.makedirs(record_json_dir)

    obs = env.reset()
    done = False

    observations = []

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
        if max(reward) > 0 and done is False:
            raise ValueError('Why?????????????????????')
        observations.append(obs)

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

    return info, steps, observations, reward


def run_tournament(tournament_name, agent_pool1, agent_pool2, match_count, AllVsAll=True, create_csv=True, get_observations=False, seed=None):
    '''Wrapper to help start the game'''
    config = 'OneVsOne-v0'
    record_pngs_dir = None #f'C:/tmp/Results/PNGS'
    record_json_dir = None #f'C:/tmp/Results/JSON'
    csv_dir = f'C:/tmp/Results/CSV'
    game_state_file = None
    render_mode = 'human'
    do_sleep = False
    render = False

    match_observations = None
    if get_observations:
        match_observations = []

    if create_csv and not os.path.isdir(csv_dir):
        os.makedirs(csv_dir)
    game_details = [['p1','p2','result','winner','time','steps', 'add_info_p1', 'add_info_p2']]
    write_csv_pos = 0

    duel_num = 0
    if AllVsAll:
        total_duels = len(agent_pool1) * len(agent_pool2)
    else:
        total_duels = len(agent_pool1)
    game_num = 0
    total_games = total_duels * match_count * 2
    tot_wins = tot_tie = tot_loss = 0
    p1_num = 0
    for p1_a in agent_pool1:
        if AllVsAll:
            tmp_agent_pool2 = agent_pool2
        else:
            tmp_agent_pool2 = [agent_pool2[p1_num]]
        p1_num += 1
        for p2_a in tmp_agent_pool2:
            duel_num += 1
            print(f'Duel {duel_num}/{total_duels}: {p1_a[0]} vs {p2_a[0]}')

            wins = ties = loss = 0
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

                m_wins = m_ties = m_loss = 0
                for i in range(match_count):
                    game_num += 1

                    record_pngs_dir_ = None
                    record_json_dir_ = None
                    if record_pngs_dir:
                        record_pngs_dir_ = f'{record_pngs_dir}/{tournament_name}/{agent_names[0]}_vs_{agent_names[1]}_{i+1}'
                    if record_json_dir:
                        record_json_dir_ = f'{record_json_dir}/{tournament_name}/{agent_names[0]}_vs_{agent_names[1]}_{i+1}'

                    start = time.time()
                    info, steps, observations, reward = run(env, agent_names, config, render, do_sleep, record_pngs_dir_, record_json_dir_)

                    if match_observations:
                        match_observations.append((observations, reward))

                    total_time = time.time() - start
                    winner = -1
                    if info['result'] == constants.Result.Win:
                        winner = int(info['winners'][0])
                        if winner == 0: m_wins += 1
                        else: m_loss += 1
                    else:
                        m_ties += 1

                    agent_info_1 = {}
                    agent_info_2 = {}
                    if isinstance(agents[0], AbstractMCTSSkeleton):
                        agents[0].get_agent_info(agent_info_1)
                    if isinstance(agents[1], AbstractMCTSSkeleton):
                        agents[1].get_agent_info(agent_info_2)

                    game_details.append([agent_names[0], agent_names[1], info['result'], winner, total_time, steps, agent_info_1, agent_info_2])

                    print(f"-- {game_num} / {total_games} Result: ", game_details[-1])

                ties += m_ties
                if d == 0:
                    wins += m_wins
                    loss += m_loss
                else:
                    wins += m_loss
                    loss += m_wins

                atexit.register(env.close)

            print(f'Result from {p1_a[0]} vs {p2_a[0]}: {wins} p1, {ties} ties, {loss} p2')
            tot_wins += wins
            tot_tie += ties
            tot_loss += loss

            if create_csv:
                f = open(f'{csv_dir}/{tournament_name}.csv', 'a')
                with f:
                    writer = csv.writer(f, delimiter=';')
                    while write_csv_pos < len(game_details):
                        writer.writerow(game_details[write_csv_pos])
                        write_csv_pos += 1

    if get_observations:
        return match_observations
    else:
        return tot_wins, tot_tie, tot_loss


def run_single_match(agent1, agent2, render=False, seed=None):
    '''Wrapper to help start the game'''
    config = 'OneVsOne-v0'
    record_pngs_dir = None #f'C:/tmp/Results/PNGS'
    record_json_dir = None #f'C:/tmp/Results/JSON'
    game_state_file = None
    render_mode = 'human'
    do_sleep = False

    agents = [agent1, agent2]

    env = make(config, agents, game_state_file, render_mode=render_mode)
    if seed is None:
        # Pick a random seed between 0 and 2^31 - 1
        seed = random.randint(0, np.iinfo(np.int32).max)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    record_pngs_dir_ = None
    record_json_dir_ = None

    info, steps, observations, reward = run(env, None, config, render, do_sleep, record_pngs_dir_, record_json_dir_)
    atexit.register(env.close)
    return reward