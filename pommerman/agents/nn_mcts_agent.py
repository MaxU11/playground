'''The MCTS skeleton'''

import math
import random
import datetime

from .abstract_mcts_agent import AbstractMCTSAgent
from .abstract_mcts_skeleton import AbstractMCTSSkeleton
from .. import characters
from .. import constants
from .env_simulator import EnvSimulator
import numpy as np

import time


EPS = 1e-8

class NN_Agent(AbstractMCTSAgent):
    """The MCTS Skeleton."""

    def __init__(self, nnet, *args, **kwargs):
        super(NN_Agent, self).__init__(character=kwargs.get('character', characters.Bomber))
        self.nnet = nnet

        # parent hyperparameter
        self.expandTreeRollout = kwargs.get('expandTreeRollout', False)
        self.maxIterations = kwargs.get('maxIterations', 1000)
        self.maxTime = kwargs.get('maxTime', 0.1)
        # hyperparameter
        self.discountFactor = kwargs.get('discountFactor', 0.9999)
        self.depthLimit = kwargs.get('depthLimit', 26)
        self.C = kwargs.get('C', 0.5) # exploration weight
        self.tempThreshold = kwargs.get('tempThreshold', 0)

        self.stopSelection = False
        self.Qsa = {}  # stores Q values for s,a
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.V = {} # stores the values (returned by neural net)

        #self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

        self.c_board = {}
        self.trainExamples = []
        self.tempCount = self.tempThreshold

    def agent_reset(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.tempCount = self.tempThreshold

        #self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

        self.c_board = {}
        self.trainExamples = []
        AbstractMCTSAgent.agent_reset(self)


    def root_changed(self, root):
        pass

    def add_train_example(self, c_board, a_probs, reward):
        sym = self.get_sym_boards(c_board, a_probs)
        for st, pt in sym:
            self.trainExamples.append((st, pt, reward))

    def act(self, obs, action_space):
        AbstractMCTSSkeleton.act(self, obs, action_space)

        s, c_board = self.get_canonical_board_str(self.root, self.root.agent_id)
        data = self.get_data(self.root)
        r = self.get_reward(self.root, data)

        if not self.root.done:
            a_probs = self.getActionProb(s, int(self.tempCount > 0))
            self.add_train_example(c_board, a_probs, r)
            action = np.random.choice(len(a_probs), p=a_probs)
        else:
            action = 0

        self.tempCount -= 1
        return action

    def get_sym_boards(self, s, a_probs):
        sym = []
        p = np.array([[None, a_probs[1], None], [a_probs[3], None, a_probs[4]], [None, a_probs[2], None]])
        for i in range(4):
            s = np.rot90(s)
            p = np.rot90(p)
            sf = np.flip(s, 0)
            pf = np.flip(p, 0)
            sym.append((s, np.array([a_probs[0], p[0][1], p[2][1], p[1][0], p[1][2], a_probs[5]])))
            sym.append((sf, np.array([a_probs[0], pf[0][1], pf[2][1], pf[1][0], pf[1][2], a_probs[5]])))
        return sym

    def create_node(self, depth, game_data, action_space, agent_id, enemy_id, action, save_data=True):
        node = AbstractMCTSAgent.create_node(self, depth, game_data, action_space, agent_id, enemy_id, action, save_data)

        s, c_board = self.get_canonical_board_str(node, node.agent_id)

        if s not in self.Ns: # initialize?
            valids = self.get_valid_actions(game_data, node.agent_id)
            self.Vs[s] = valids

            nn_input = self.nnet.get_nn_input(c_board)
            p, v = self.nnet.predict(nn_input)

            valids = self.Vs[s]
            p = p * valids  # masking invalid moves

            sum_Ps_s = np.sum(p)
            if sum_Ps_s > 0:
                p /= sum_Ps_s  # renormalize
            else:
                print("All valid moves were masked, doing a workaround. Overfitting?")
                p = s + valids
                p /= np.sum(p)

            self.Ps[s] = p
            self.V[s] = v
            self.Ns[s] = 0
        else:
            valids = self.Vs[s]

        node.unseen_actions = [a for a in range(self.action_space) if valids[a]]
        return node

    def get_selected_child(self, node):
        # select child for traversing using UCB
        action = self.get_ucb_action(node)
        return node.children[action]

    def get_ucb_action(self, node):
        s, _ = self.get_canonical_board_str(node, node.agent_id)
        valids = self.Vs[s]

        a_best_v = 0
        ucp_val_v = -float('inf')
        a_best_c = 0
        ucp_val_c = -float('inf')
        for a in range(self.action_space):
            if valids[a]:
                if (s, a) in self.Qsa:
                    v = self.Qsa[(s, a)]
                    cpar = self.C * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    a_best_v = None
                    ucp_val_v = None
                    cpar = self.C * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)

                if ucp_val_v is not None and (v + cpar) > ucp_val_v:
                    a_best_v = a
                    ucp_val_v = (v + cpar)
                if cpar > ucp_val_c:
                    a_best_c = a
                    ucp_val_c = cpar

        if a_best_v is not None:
            return a_best_v
        else:
            return a_best_c

    def non_terminal_rollout(self, node):
        # nn agent will evaluate selected child directly
        return False

    def non_terminal(self, node):
        # check if node is terminal
        if self.depthLimit and (node.depth - self.root.depth) >= self.depthLimit:
            return False

        return AbstractMCTSAgent.non_terminal(self, node)

    def get_my_expand_action(self, node):
        return random.choice(node.unseen_actions)

    def get_enemy_expand_action(self, node):
        return random.choice(node.unseen_actions)

    def get_my_rollout_action(self, node, data):
        raise ValueError('should do no rollout')

    def get_enemy_rollout_action(self, node, data):
        raise ValueError('should do no rollout')

    def result(self, node, data):
        if node.done:
            return self.get_reward(node, data)

        s, _ = self.get_canonical_board_str(node, node.agent_id)
        v = self.V[s]

        return [v, None]

    def get_reward(self, node, data):
        if node.agent_id != self.agent_id:
            raise ValueError('why reward from enemy?')

        # get reward from terminal node
        reward = 0.0
        alive = EnvSimulator.get_alive(data)
        me_alive = False
        enemy_alive = False
        for a in alive:
            if a == self.agent_id:
                me_alive = alive[a]
            else:
                enemy_alive = alive[a]

        if me_alive and enemy_alive:
            reward = [0, 0]
        elif me_alive and not enemy_alive:
            reward = [1, -1]
        elif not me_alive and enemy_alive:
            reward = [-1, 1]
        elif not me_alive and not enemy_alive:
            reward = [-1, -1]

        return reward

    def update_stats(self, node, action, result):
        if not node.done and action is not None: # action is none for expanded node -> we store Q(s, a) -> skip leaf node
            s, c_board = self.get_canonical_board_str(node, node.agent_id)

            if node.agent_id == self.agent_id:
                r = result[0]
            else:
                r = result[1]

            if r != None:
                # get updated node stats
                if (s, action) in self.Qsa:
                    self.Qsa[(s, action)] = (self.Nsa[(s, action)] * self.Qsa[(s, action)] + r) / (self.Nsa[(s, action)] + 1)
                    self.Nsa[(s, action)] += 1
                else:
                    self.Qsa[(s, action)] = r
                    self.Nsa[(s, action)] = 1

                self.Ns[s] += 1

                if node.agent_id == self.agent_id:
                    result[0] *= self.discountFactor
                else:
                    result[1] *= self.discountFactor
            else:
                result[1] = self.V[s]

        return result

    def best_child(self, node):
        return 0

    def getActionProb(self, s_board, temp):
        counts = [self.Nsa[(s_board, a)] if (s_board, a) in self.Nsa else 0 for a in range(self.action_space)]
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs
        else:
            counts = [x ** (1. / temp) for x in counts]
            counts_sum = float(sum(counts))
            probs = [x / counts_sum for x in counts]
            return probs

    def get_canonical_board_str(self, node, agent_id):
        if node in self.c_board:
            return self.c_board[node]
        str_rep, c_board = NN_Agent.get_canonical_board_str_from_data(self.get_data(node), agent_id)
        self.c_board[node] = (str_rep, c_board)
        return str_rep, c_board

    @staticmethod
    def get_canonical_board_str_from_data(data, agent_id):
        c_board = NN_Agent.get_canonical_board_from_data(data, agent_id)
        str_rep = ""
        for row in range(c_board.shape[0]):
            for col in range(c_board.shape[1]):
                str_rep += str(c_board[row, col]) + ','
            str_rep += '\n'
        return str_rep, c_board

    def get_canonical_board(self, node, agent_id):
        return NN_Agent.get_canonical_board_from_data(self.get_data(node), agent_id)

    @staticmethod
    def get_canonical_board_from_data(data, agent_id):
        c_board = np.zeros(data.board.shape, dtype=np.int32)
        for row in range(data.board.shape[0]):
            for col in range(data.board.shape[1]):
                val = data.board[row, col]
                if val == constants.Item.Rigid.value:
                    c_board[row, col] += 1 << 0
                elif val == constants.Item.Wood.value:
                    c_board[row, col] += 1 << 1
                elif val == constants.Item.ExtraBomb.value or val == constants.Item.IncrRange.value or val == constants.Item.Kick.value:
                    c_board[row, col] += 1 << 2
                elif val == 10:
                    if agent_id == 0: c_board[row, col] += 1 << 3
                    else: c_board[row, col] += 1 << 4
                elif val == 11:
                    if agent_id == 1: c_board[row, col] += 1 << 3
                    else: c_board[row, col] += 1 << 4
        for b in data.bombs:
            if b.blast_strength > 7:
                raise ValueError('invalid blast strength!')
            c_board[b.position] += (10 - b.life) << 5
            c_board[b.position] += b.blast_strength << 9
        for f in data.flames:
            c_board[f.position] += f.life << 12

        return c_board

    def get_valid_actions(self, game_data, agent_id):
        # check valid actions
        valid_actions = [False] * self.action_space
        for agent in game_data.agents:
            if agent.agent_id is agent_id:
                if agent.is_alive:
                    v_actions = EnvSimulator.get_valid_actions(game_data.board, game_data.flames, game_data.bombs, agent, list(range(self.action_space)))
                else:
                    v_actions = [constants.Action.Stop.value]
                if len(v_actions) == 0:
                    #print(agent_id, 'good bye')
                    v_actions = [constants.Action.Stop.value]

                for a in v_actions:
                    valid_actions[a] = True
                return valid_actions

        raise ValueError('Invalid agent id')

    def get_agent_map_info(self, node):
        return ''