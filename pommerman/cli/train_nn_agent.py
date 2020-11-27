from pommerman.NN.Coach import Coach
from pommerman.NN.pommerman_neural_net import PommermanNNet
from pommerman import constants
import torch

args = {'numIters': 1000,
        'numEps': 2,              # Number of complete self-play games to simulate during a new iteration.
        'tempThreshold': 15,        #
        'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
        'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 1,

        'log_folder': 'c:\\tmp\\Model',
        'log_file': 'train.log',
        'loss_file': 'loss.csv',
        'checkpoint_folder': 'c:\\tmp\\Model',
        'load_model': False,
        'load_folder_file': ('c:\\tmp\\Model\\','best.pth.tar'),
        'numTrainExamplesHistory': 20,
        'df': 1,

        'expandTreeRollout': False,
        'maxIterations': 25,
        'maxTime': 0.0,
        'discountFactor': 0.9999,
        'depthLimit': 26,
        'C': 0.5,
        'temp': 1
        }

nn_args = {
    'input_channels': 8,
    'board_x': constants.BOARD_SIZE_ONE_VS_ONE,
    'board_y': constants.BOARD_SIZE_ONE_VS_ONE,

    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 0, # 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512
}

def main():
    nnet = PommermanNNet(**nn_args)
    nnet.id = 1

    if args['load_model']:
        print('Loading checkpoint "%s/%s"...', args['load_folder_file'])
        nnet.load_checkpoint(args['load_folder_file'][0], args['load_folder_file'][1])
    else:
        print('Not loading a checkpoint!')

    coach = Coach(nnet, args, nn_args)

    if args['load_model']:
        print("Loading 'trainExamples' from file...")
        coach.loadTrainExamples()

    coach.make_training(args['numIters'], args['numEps'])


if __name__ == "__main__":
    main()
