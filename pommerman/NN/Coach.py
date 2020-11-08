from collections import deque
from tqdm import tqdm
from random import shuffle

from ..cli.Tournament import run_tournament
from ..cli.Tournament import run_single_match
from pommerman.agents.nn_mcts_agent import NN_Agent

class Coach:

    def __init__(self, nnet, updateThreshold, *args, **kwargs):
        self.nnet = nnet
        self.updateThreshold = updateThreshold
        self.checkpoint_folder = 'c:\\tmp\\Model'
        self.cur_iteration = 0

        # Agnets hyperparameter
        self.a_kwds = {'expandTreeRollout': False,
                       'maxIterations': 1000,
                       'maxTime': 0.1,
                       'discountFactor': 0.9999,
                       'depthLimit': 26,
                       'C': 0.5,
                       'temp': 1}

    def make_training(self, numIters, numEps, numTrainExamplesHistory, threshold, df):
        nnet = self.initNNet()  # initialise random neural network
        trainExamplesHistory = []
        for i in range(numIters):
            self.cur_iteration += 1

            for _ in tqdm(range(numEps), desc="Self Play"):
                trainExamplesHistory.append(self.executeEpisode(nnet, df))  # collect examples from this game

            if len(trainExamplesHistory) > numTrainExamplesHistory:
                trainExamplesHistory.pop(0)

            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            prev_net = self.copyNNet(nnet)
            self.trainNNet(nnet, trainExamples)
            self.validate_nn(nnet, prev_net)

        return nnet

    def initNNet(self):
        return self.nnet

    def executeEpisode(self, nnet, df):
        agent1 = NN_Agent(nnet)
        agent2 = NN_Agent(nnet)

        reward = run_single_match(agent1, agent2)

        trainExamples_p1 = agent1.trainExamples
        trainExamples_p2 = agent1.trainExamples

        if (len(trainExamples_p1) != len(trainExamples_p2)):
            raise ValueError('obs should have same amount.')
        if (len(trainExamples_p1) % 4) != 0:
            raise ValueError('invalid obs length')

        r1 = reward[0]
        r2 = reward[1]
        for i in reversed(range(len(trainExamples_p1))):
            if i > 0 and i % 4 == 0:
                r1 *= df
                r2 *= df
            trainExamples_p1[i] = r1
            trainExamples_p2[i] = r2

        return trainExamples_p1 + trainExamples_p2

    def copyNNet(self, nnet):
        nnet.save_checkpoint(folder=self.checkpoint_folder, filename='temp.pth.tar')
        copy_nnet = nnet.__class__()
        copy_nnet.load_checkpoint(folder=self.checkpoint_folder, filename='temp.pth.tar')

        return copy_nnet

    def trainNNet(self, nnet, examples):
        nnet.train(examples)

    def validate_nn(self, nnet, prev_nnet):
        agent_pool1 = [NN_Agent(nnet, **self.a_kwds)]
        agent_pool2 = [NN_Agent(prev_nnet, **self.a_kwds)]
        match_count = 50
        wins, ties, loss = run_tournament('tournament_name', agent_pool1, agent_pool2, match_count, False, False, False)

        print(f'compare results: {wins} wins, {ties} ties, {loss} loss')
        if wins + loss == 0 or float(wins) / (wins + loss) < self.updateThreshold:
            print('REJECTING NEW MODEL')
            nnet.load_checkpoint(folder=self.checkpoint_folder, filename='temp.pth.tar')
        else:
            print('ACCEPTING NEW MODEL')
            nnet.save_checkpoint(folder=self.checkpoint_folder, filename='checkpoint_' + str(self.cur_iteration) + '.pth.tar')
            nnet.save_checkpoint(folder=self.checkpoint_folder, filename='best.pth.tar')