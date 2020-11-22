'''The MCTS skeleton'''

import math

from .abstract_mcts_agent import AbstractMCTSAgent
from .abstract_mcts_skeleton import AbstractMCTSSkeleton
from .. import characters
from .. import constants
from .env_simulator import EnvSimulator
import numpy as np


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
        self.temp = kwargs.get('temp', 1)
        self.stopSelection = False

        self.Qsa = {}  # stores Q values for s,a
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        #self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def agent_reset(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        #self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        AbstractMCTSAgent.agent_reset(self)

        self.trainExamples = []

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
        a_probs = self.getActionProb(s)

        self.add_train_example(c_board, a_probs, r)

        action = np.random.choice(len(a_probs), p=a_probs)
        return action

    def get_sym_boards(self, s, a_probs):
        sym = []
        p = np.array([a_probs[1], a_probs[3], a_probs[2], a_probs[4]])
        for i in range(4):
            s = np.rot90(s)
            p = np.rot90(p)
            sf = np.flip(s, 0)
            pf = np.flip(p, 0)
            sym.append((s, np.array([p[0], p[1], p[3], p[2], p[4], p[5]])))
            sym.append((sf, np.array([pf[0], pf[1], pf[3], pf[2], pf[4], pf[5]])))
        return sym

    def get_ucb_action(self, node):
        s, _ = self.get_canonical_board_str(node, node.agent_id)
        valids = self.Vs[s]

        a_s = []
        ucp_val = []
        for a in range(self.action_space):
            if valids[a]:
                if (s, a) in self.Qsa:
                    v = self.Qsa[(s, a)]
                    cpar = self.C * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    v = 0
                    cpar = self.C * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)

                if node.agent_id is self.agent_id:
                    a_s.append(a)
                    ucp_val.append(v + cpar)
                else:
                    a_s.append(a)
                    ucp_val.append(v - cpar)

        if node.agent_id is self.agent_id:
            return a_s[ucp_val.index(max(ucp_val))]
        else:
            return a_s[ucp_val.index(min(ucp_val))]

    def node_fully_expanded(self, node):
        if self.stopSelection:
            self.stopSelection = False
            return False

        s, _ = self.get_canonical_board_str(node, node.agent_id)
        return (s in self.Ps)

    def non_terminal_traverse(self, node):
        # nn agent will evaluate selected child directly
        return False

    def non_terminal_rollout(self, node):
        # nn agent will evaluate selected child directly
        return False

    def get_selected_child(self, node):
        # select child for traversing using UCB
        action = self.get_ucb_action(node)

        if not action in node.children:
            game_data = self.get_data(node)
            self.expand_node(node, game_data, action)
            self.stopSelection = True

        return node.children[action]

    def get_my_expand_action(self, node):
        raise ValueError('should do no expand')

    def get_enemy_expand_action(self, node):
        raise ValueError('should do no expand')

    def get_my_rollout_action(self, node, data):
        raise ValueError('should do no rollout')

    def get_enemy_rollout_action(self, node, data):
        raise ValueError('should do no rollout')

    def result(self, node, data):
        if node.done:
            return self.get_reward(node, data)

        s, c_board = self.get_canonical_board_str(node, node.agent_id)
        nn_input = self.get_nn_input(c_board)
        self.Ps[s], v = self.nnet.predict(nn_input)

        if s in self.Vs:
            valids = self.Vs[s]
        else:
            valids = self.get_valid_actions(data, node.agent_id)
        self.Ps[s] = self.Ps[s] * valids  # masking invalid moves

        sum_Ps_s = np.sum(self.Ps[s])
        if sum_Ps_s > 0:
            self.Ps[s] /= sum_Ps_s  # renormalize
        else:
            # if all valid moves were masked make all valid moves equally probable
            # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
            # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
            print("All valid moves were masked, doing a workaround.")
            self.Ps[s] = self.Ps[s] + valids
            self.Ps[s] /= np.sum(self.Ps[s])

        self.Vs[s] = valids
        self.Ns[s] = 0
        return v

    def get_reward(self, node, data):
        # get reward from terminal node
        reward = 0.0
        alive = EnvSimulator.get_alive(data)
        for a in alive:
            if a == self.agent_id:
                if alive[a]:
                    reward += 1.0
                else:
                    reward += -1.0
            else:
                if alive[a]:
                    reward += -0.5
                else:
                    reward += 0.5
        return reward

    def update_stats(self, node, action, result):
        s, c_board = self.get_canonical_board_str(node, node.agent_id)

        # get updated node stats
        if (s, action) in self.Qsa:
            self.Qsa[(s, action)] = (self.Nsa[(s, action)] * self.Qsa[(s, action)] + result) / (self.Nsa[(s, action)] + 1)
            self.Nsa[(s, action)] += 1
        else:
            self.Qsa[(s, action)] = result
            self.Nsa[(s, action)] = 1

        self.Ns[s] += 1
        return result * self.discountFactor

    def best_child(self, node):
        return 0

    def getActionProb(self, s_board):
        counts = [self.Nsa[(s_board, a)] if (s_board, a) in self.Nsa else 0 for a in range(self.action_space)]
        counts = [x ** (1. / self.temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]

        return probs

    def get_nn_input(self, c_board):
        rigid_layer = self.binary_filter(c_board, 0)
        wood_layer = self.binary_filter(c_board, 1)
        item_layer = self.binary_filter(c_board, 2)
        me_layer = self.binary_filter(c_board, 3)
        enemy_layer = self.binary_filter(c_board, 4)
        blife_layer = self.binary_filter(c_board, 5, 4)
        bstrength_layer = self.binary_filter(c_board, 9, 3)
        flames_layer = self.binary_filter(c_board, 12, 2)
        return np.array([rigid_layer, wood_layer, item_layer, me_layer, enemy_layer, blife_layer, bstrength_layer, flames_layer])

    def binary_filter(self, layer, pos, l=1):
        n = ((2 ** l) - 1) << pos
        mask = np.ones(layer.shape, dtype=np.int32) * n
        ml = np.bitwise_and(layer, mask)
        return np.right_shift(ml, pos)

    def get_canonical_board_str(self, node, agent_id):
        c_board = self.get_canonical_board(node, agent_id)
        str_rep = ""
        for row in range(c_board.shape[0]):
            for col in range(c_board.shape[1]):
                str_rep += str(c_board[row, col]) + ','
            str_rep += '\n'
        return str_rep, c_board

    def get_canonical_board(self, node, agent_id):
        data = self.get_data(node)
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