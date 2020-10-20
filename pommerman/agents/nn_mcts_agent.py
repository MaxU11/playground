'''The MCTS skeleton'''

import datetime

from abc import ABC, abstractmethod
from . import BaseAgent
from .. import characters


class NN_Agent(ABC, BaseAgent):
    """The MCTS Skeleton."""

    def __init__(self, nnet, *args, **kwargs):
        super(NN_Agent, self).__init__(character=kwargs.get('character', characters.Bomber))
        self.nnet = nnet

    def reset(self):
        pass

    def act(self, obs, action_space):
