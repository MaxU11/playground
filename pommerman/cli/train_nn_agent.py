from collections import deque

def policyIterSP(game):
    nnet = initNNet()  # initialise random neural network
    examples = []
    for i in range(numIters):
        for e in range(numEps):
            examples += executeEpisode(game, nnet)  # collect examples from this game
        new_nnet = trainNNet(examples)
        frac_win = pit(new_nnet, nnet)  # compare new net with previous net
        if frac_win > threshold:
            nnet = new_nnet  # replace with new net
    return nnet


def executeEpisode(game, nnet):
    examples = []
    s = game.startState()
    mcts = MCTS()  # initialise search tree

    while True:
        for _ in range(numMCTSSims):
            mcts.search(s, game, nnet)
        examples.append([s, mcts.pi(s), None])  # rewards can not be determined yet
        a = random.choice(len(mcts.pi(s)), p=mcts.pi(s))  # sample action from improved policy
        s = game.nextState(s, a)
        if game.gameEnded(s):
            examples = assignRewards(examples, game.gameReward(s))
            return examples

def learn(self):

    for i in range(1, self.args.numIters + 1):
        # bookkeeping
        print(f'Starting Iter #{i} ...')
        # examples of the iteration
        if not self.skipFirstSelfPlay or i > 1:
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                iterationTrainExamples += self.executeEpisode()

            # save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)

        if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
            log.warning(
                f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
            self.trainExamplesHistory.pop(0)
        # backup history to a file
        # NB! the examples were collected using the model from the previous iteration, so (i-1)
        self.saveTrainExamples(i - 1)

        # shuffle examples before training
        trainExamples = []
        for e in self.trainExamplesHistory:
            trainExamples.extend(e)
        shuffle(trainExamples)

        # training new network, keeping a copy of the old one
        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
        self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
        pmcts = MCTS(self.game, self.pnet, self.args)

        self.nnet.train(trainExamples)
        nmcts = MCTS(self.game, self.nnet, self.args)

        log.info('PITTING AGAINST PREVIOUS VERSION')
        arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                      lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
        pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

        log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
        if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
            log.info('REJECTING NEW MODEL')
            self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
        else:
            log.info('ACCEPTING NEW MODEL')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
