import torch
import os

from pommerman.NN.Coach import Coach
from pommerman.NN.pommerman_neural_net import PommermanNNet
from pommerman import constants

args = {'numIters': 1000,
        'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
        'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
        'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 1,

        'folder': 'c:\\tmp\\Model',
        'log_file': 'train.log',
        'loss_file': 'loss.csv',
        'load_model': False,
        'skip_first_play': False,
        'checkpoint_folder_file': ('c:\\tmp\\Model\\','best.pth.tar'),
        'examples_folder_file': ('c:\\tmp\\Model\\','best.pth.tar.examples'),
        'numTrainExamplesHistory': 20,
        'df': 1,

        'expandTreeRollout': False,
        'maxIterations': 25,
        'maxTime': 0.0,
        'discountFactor': 0.9999,
        'depthLimit': 26,
        'C': 1.0,
        'tempThreshold': 5
        }

nn_args = {
    'input_channels': 8,
    'board_x': constants.BOARD_SIZE_ONE_VS_ONE,
    'board_y': constants.BOARD_SIZE_ONE_VS_ONE,

    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10, # 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512
}

def main():
    nnet = PommermanNNet(**nn_args)
    nnet.id = 1

    if args['load_model']:
        print('Loading checkpoint "%s/%s"...', args['checkpoint_folder_file'])
        nnet.load_checkpoint(args['checkpoint_folder_file'][0], args['checkpoint_folder_file'][1])
    else:
        print('Not loading a checkpoint!')

    coach = Coach(nnet, args, nn_args)

    if args['load_model']:
        print("Loading 'trainExamples' from file...", args['examples_folder_file'])
        coach.loadTrainExamples(args['examples_folder_file'][0], args['examples_folder_file'][1], args['skip_first_play'])

    coach.make_training(args['numIters'], args['numEps'])


if __name__ == "__main__":
    main()
