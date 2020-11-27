import os
import sys
import numpy as np
import datetime

from pickle import Pickler, Unpickler
from collections import deque
from tqdm import tqdm
from random import shuffle

from ..cli.Tournament import run_tournament
from ..cli.Tournament import run_single_match
from pommerman.agents.nn_mcts_agent import NN_Agent

import logging

class Coach:

    def __init__(self, nnet, args, nn_args):
        log_file = args.get('log_folder', 'c:/tmp/Model')
        log_file += '/' + args.get('log_file', 'train.log')
        logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(message)s')

        self.nnet = nnet
        self.updateThreshold = args.get('updateThreshold', 0.6)
        self.checkpoint_folder = args.get('checkpoint_folder', 'c:/tmp/Model')

        self.numTrainExamplesHistory = args.get('numTrainExamplesHistory', 20)
        self.maxlenOfQueue = args.get('maxlenOfQueue', 200000)
        self.df = args.get('df', 1.0)

        self.cur_iteration = 0
        self.args = args
        self.nn_args = nn_args

        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

        # Agnets hyperparameter
        self.a_kwds = {'expandTreeRollout': args.get('expandTreeRollout', False),
                       'maxIterations': args.get('maxIterations', 1000),
                       'maxTime': args.get('maxTime', 0.0),
                       'discountFactor': args.get('discountFactor', 0.9999),
                       'depthLimit': args.get('depthLimit', 26),
                       'C': args.get('C', 0.5),
                       'temp': args.get('temp', 1)}

    def make_training(self, numIters, numEps):

        nnet = self.initNNet()  # initialise random neural network
        for i in range(numIters):
            self.cur_iteration += 1

            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples  = deque([], maxlen=self.maxlenOfQueue)

                logging.info(f'nnet {nnet.id}: start execute episodes')
                for _ in tqdm(range(numEps), desc="Self Play"):
                    iterationTrainExamples  += self.executeEpisode(nnet, self.df)  # collect examples from this game
                logging.info(f'nnet {nnet.id}: finished execute episodes')

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.numTrainExamplesHistory:
                self.trainExamplesHistory.pop(0)

            self.saveTrainExamples(i - 1)

            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            prev_net = self.copyNNet(nnet)
            nnet.id += 1
            logging.info(f'nnet {prev_net.id} -> nnet {nnet.id}')
            logging.info(f'nnet {nnet.id}: start training')
            self.trainNNet(nnet, trainExamples)
            logging.info(f'nnet {nnet.id}: validate nnet')
            self.validate_nn(nnet, prev_net)

        return nnet

    def initNNet(self):
        return self.nnet

    def executeEpisode(self, nnet, df):
        agent1 = NN_Agent(nnet, **self.a_kwds)
        agent2 = NN_Agent(nnet, **self.a_kwds)

        start_t = datetime.datetime.now()
        reward = run_single_match(agent1, agent2)

        trainExamples_p1 = agent1.trainExamples
        trainExamples_p2 = agent1.trainExamples

        if (len(trainExamples_p1) != len(trainExamples_p2)):
            raise ValueError('obs should have same amount.')
        if (len(trainExamples_p1) % 8) != 0:
            raise ValueError('invalid obs length')

        j=0
        r1 = reward[0]
        r2 = reward[1]
        for i in reversed(range(len(trainExamples_p1))):
            if j > 0 and j % 8 == 0:
                r1 *= df
                r2 *= df
            j += 1
            trainExamples_p1[i] = (trainExamples_p1[i][0], trainExamples_p1[i][1], r1)
            trainExamples_p2[i] = (trainExamples_p2[i][0], trainExamples_p2[i][1], r2)

        return trainExamples_p1 + trainExamples_p2

    def copyNNet(self, nnet):
        nnet.save_checkpoint(folder=self.checkpoint_folder, filename='temp.pth.tar')
        copy_nnet = nnet.__class__(**self.nn_args)
        copy_nnet.load_checkpoint(folder=self.checkpoint_folder, filename='temp.pth.tar')

        copy_nnet.id = nnet.id
        return copy_nnet

    def trainNNet(self, nnet, examples):
        nnet.make_train(examples, folder=self.args['log_folder'], filename=self.args['loss_file'])

    def validate_nn(self, nnet, prev_nnet):
        agent_pool1 = [(f'NN{nnet.id}_Agent', NN_Agent, {'nnet':nnet, **self.a_kwds})]
        agent_pool2 = [(f'NN{prev_nnet.id}_Agent', NN_Agent, {'nnet':prev_nnet, **self.a_kwds})]
        match_count = self.args['arenaCompare']
        wins, ties, loss = run_tournament('tournament_name', agent_pool1, agent_pool2, int(match_count/2), False, False, False)

        logging.info(f'nnet {nnet.id} X nnet {prev_nnet.id}: {wins} wins, {ties} ties, {loss} loss')
        #print(f'compare results: {wins} wins, {ties} ties, {loss} loss')
        if wins + loss == 0 or float(wins) / (wins + loss) < self.updateThreshold:
            #print('REJECTING NEW MODEL')
            logging.info(f'REJECTING nnet {nnet.id} <- nnet {prev_nnet.id}')
            nnet.load_checkpoint(folder=self.checkpoint_folder, filename='temp.pth.tar')
            nnet.id = prev_nnet.id
            return False
        else:
            #print('ACCEPTING NEW MODEL')
            logging.info(f'ACCEPTING nnet {nnet.id}')
            nnet.save_checkpoint(folder=self.checkpoint_folder, filename=self.getCheckpointFile(self.cur_iteration))
            nnet.save_checkpoint(folder=self.checkpoint_folder, filename='best.pth.tar')
            return True

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.checkpoint_folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args['load_folder_file'][0], self.args['load_folder_file'][1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            logging.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            logging.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            logging.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True