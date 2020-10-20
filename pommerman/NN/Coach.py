from collections import deque
from tqdm import tqdm

from ..cli.Tournament import run_tournament
from pommerman.agents.nn_mcts_agent import

class Coach:

    def __init__(self, nnet, updateThreshold, *args, **kwargs):
        self.nnet = nnet
        self.updateThreshold = updateThreshold
        self.checkpoint_folder = 'c:\\tmp\\Model'
        self.cur_iteration = 0

    def make_training(self, numIters, numEps, numTrainExamplesHistory, threshold, df):
        nnet = self.initNNet()  # initialise random neural network
        trainExamplesHistory = []
        for i in range(numIters):
            self.cur_iteration += 1

            for _ in tqdm(range(numEps), desc="Self Play"):
                trainExamplesHistory.append(self.executeEpisode(nnet, df))  # collect examples from this game

            if len(trainExamplesHistory) > numTrainExamplesHistory:
                trainExamplesHistory.pop(0)

            prev_net = self.copyNNet(nnet)
            self.trainNNet(nnet, trainExamplesHistory)
            self.validate_nn(nnet, prev_net)

        return nnet

    def initNNet(self):
        return self.nnet

    def executeEpisode(self, nnet, df):
        agent_pool1 = []
        agent_pool2 = []
        match_count = 1
        match_observations = run_tournament('tournament_name', agent_pool1, agent_pool2, match_count, False, False, True)

        examples = []
        for observations, reward in match_observations:
            for i in range(len(observations)):
                examples.append((observations[-1 - i], reward))
                reward *= df

        return examples

    def copyNNet(self, nnet):
        nnet.save_checkpoint(folder=self.checkpoint_folder, filename='temp.pth.tar')
        copy_nnet = nnet.__class__()
        copy_nnet.load_checkpoint(folder=self.checkpoint_folder, filename='temp.pth.tar')

        return copy_nnet

    def trainNNet(self, nnet, examples):
        nnet.train(examples)

    def validate_nn(self, nnet, prev_nnet):
        agent_pool1 = []
        agent_pool2 = []
        match_count = 100
        wins, ties, loss = run_tournament('tournament_name', agent_pool1, agent_pool2, match_count, False, False, False)

        print(f'compare results: {wins} wins, {ties} ties, {loss} loss')
        if wins + loss == 0 or float(wins) / (wins + loss) < self.updateThreshold:
            print('REJECTING NEW MODEL')
            nnet.load_checkpoint(folder=self.checkpoint_folder, filename='temp.pth.tar')
        else:
            print('ACCEPTING NEW MODEL')
            nnet.save_checkpoint(folder=self.checkpoint_folder, filename='checkpoint_' + str(self.cur_iteration) + '.pth.tar')
            nnet.save_checkpoint(folder=self.checkpoint_folder, filename='best.pth.tar')