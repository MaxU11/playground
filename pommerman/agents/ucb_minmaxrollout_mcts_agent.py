# The UCB MiniMax MCTS agent

from collections import defaultdict
from .ucb_mcts_agent import UcbMCTSAgent


class UcbMinMaxRolloutMCTSAgent(UcbMCTSAgent):
    """The MiniMaxRollout-MCTS Agent."""

    def __init__(self, *args, **kwargs):
        super(UcbMinMaxRolloutMCTSAgent, self).__init__(*args, **kwargs)
        self.maxIterations = 1000
        self.iterations = 0
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.C = 0.5  # exploration weight
        self.discount_factor = 0.9999
        self.depth_limit = 26
        self.expand_tree_rollout = False

