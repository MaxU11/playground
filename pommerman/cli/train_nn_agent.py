from pommerman.NN.Coach import Coach
from pommerman.NN.pommerman_neural_net import PommermanNNet
from pommerman import constants

args = {'numIters': 1000,
        'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
        'tempThreshold': 15,        #
        'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
        'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
        'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 1,

        'checkpoint_folder': 'c\tmp\Model',
        'load_model': False,
        'load_folder_file': ('c/tmp/Model/','best.pth.tar'),
        'numTrainExamplesHistory': 20,
        'df': 10,

        'expandTreeRollout': False,
        'maxIterations': 1000,
        'maxTime': 0.0,
        'discountFactor': 0.9999,
        'depthLimit': 26,
        'C': 0.5,
        'temp': 1
        }

def main():
    nnet = PommermanNNet(8, constants.BOARD_SIZE_ONE_VS_ONE, constants.BOARD_SIZE_ONE_VS_ONE)

    if args['load_model']:
        print('Loading checkpoint "%s/%s"...', args['load_folder_file'])
        nnet.load_checkpoint(args['load_folder_file'][0], args['load_folder_file'][1])
    else:
        print('Not loading a checkpoint!')

    #if args['load_model']:
    #    print("Loading 'trainExamples' from file...")
    #    coach.loadTrainExamples()

    coach = Coach(nnet, args)

    coach.make_training(args['numIters'], args['numEps'])


if __name__ == "__main__":
    main()
