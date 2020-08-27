from pommerman.cli import run_battle
import argparse

def main():
    '''CLI entry pointed used to bootstrap a battle'''
    simple_agent = 'test::agents.UBC_MCTSAgent'
    player_agent = 'playerblock::arrows'
    #docker_agent = 'docker::pommerman/simple-agent'

    parser = argparse.ArgumentParser(description='Playground Flags.')
    parser.add_argument(
        '--config',
        default='OneVsOne-v0',
        help='Configuration to execute. See env_ids in '
        'configs.py for options.')
    parser.add_argument(
        '--agents',
        default=','.join([simple_agent] + [player_agent]),
        # default=','.join([player_agent] + [simple_agent]*3]),
        # default=','.join([docker_agent] + [simple_agent]*3]),
        help='Comma delineated list of agent types and docker '
        'locations to run the agents.')
    parser.add_argument(
        '--agent_env_vars',
        help='Comma delineated list of agent environment vars '
        'to pass to Docker. This is only for the Docker Agent.'
        " An example is '0:foo=bar:baz=lar,3:foo=lam', which "
        'would send two arguments to Docker Agent 0 and one '
        'to Docker Agent 3.',
        default="")
    parser.add_argument(
        '--record_pngs_dir',
        default=None,
        help='Directory to record the PNGs of the game. '
        "Doesn't record if None.")
    parser.add_argument(
        '--record_json_dir',
        default=None,
        help='Directory to record the JSON representations of '
        "the game. Doesn't record if None.")
    parser.add_argument(
        "--render",
        default=True,
        action='store_true',
        help="Whether to render or not. Defaults to False.")
    parser.add_argument(
        '--render_mode',
        default='human',
        help="What mode to render. Options are human, rgb_pixel, and rgb_array")
    parser.add_argument(
        '--game_state_file',
        default=None,
        help="File from which to load game state.")
    parser.add_argument(
        '--do_sleep',
        default=True,
        help="Whether we sleep after each rendering.")
    args = parser.parse_args()
    run_battle.run(args)

if __name__ == "__main__":
    main()