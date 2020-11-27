import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class PommermanNNet(nn.Module):

    def __init__(self, input_channels, board_x, board_y, **kwargs):
        super(PommermanNNet, self).__init__()

        self.id = 0

        self.input_channels = input_channels
        self.board_x = board_x
        self.board_y = board_y
        self.lr = kwargs.get('lr', 0.001)
        self.dropout = kwargs.get('dropout', 0.3)
        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 64)
        self.use_cuda = kwargs.get('cuda', torch.cuda.is_available())
        self.num_channels = kwargs.get('num_channels', 512)

        # game params
        self.action_size = 6

        super(PommermanNNet, self).__init__()
        self.conv1 = self.Conv2d(input_channels, self.num_channels, 3, stride=1, padding=1)
        self.conv2 = self.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1)
        self.conv3 = self.Conv2d(self.num_channels, self.num_channels, 3, stride=1)
        self.conv4 = self.Conv2d(self.num_channels, self.num_channels, 3, stride=1)

        self.bn1 = self.BatchNorm2d(self.num_channels)
        self.bn2 = self.BatchNorm2d(self.num_channels)
        self.bn3 = self.BatchNorm2d(self.num_channels)
        self.bn4 = self.BatchNorm2d(self.num_channels)

        self.fc1 = self.Linear(self.num_channels * self.board_x * self.board_y, 1024)
        self.fc_bn1 = self.BatchNorm1d(1024)

        self.fc2 = self.Linear(1024, 512)
        self.fc_bn2 = self.BatchNorm1d(512)

        self.fc3 = self.Linear(512, self.action_size)

        self.fc4 = self.Linear(512, 1)

        if self.use_cuda:
            self.cuda()

    def Conv2d(self, input_channels, num_channels, kernel_size, stride=1, padding=1):
        conv = nn.Conv2d(input_channels, num_channels, kernel_size, stride=stride, padding=padding)
        #torch.nn.init.xavier_uniform(conv.weight)
        return conv

    def BatchNorm2d(self, num_channels):
        bn = nn.BatchNorm2d(num_channels)
        #torch.nn.init.xavier_uniform(bn.weight)
        return bn

    def BatchNorm1d(self, num_channels):
        bn = nn.BatchNorm1d(num_channels)
        #torch.nn.init.xavier_uniform(bn.weight)
        return bn

    def Linear(self, in_size, out_size):
        lin = nn.Linear(in_size, out_size)
        #torch.nn.init.xavier_uniform(lin.weight)
        return lin

    # Applies forward propagation to the inputs
    def forward(self, s):

        #s = s.view(-1, -1, self.board_x, self.board_y)  # batch_size x input_channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))  # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))  # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.num_channels * self.board_x * self.board_y)

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def predict(self, board):
        """
        board: np array with board
        """

        #return np.random.uniform(0, 1, size=self.action_size), np.random.uniform(0, 1, size=1)

        # preparing input
        board = torch.FloatTensor(board.astype(np.int16))
        if self.use_cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.input_channels, self.board_x, self.board_y) # batchsize = 1
        self.eval()
        with torch.no_grad():
            pi, v = self(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def make_train(self, examples, folder='checkpoint', filename='loss.csv'):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        if not os.path.isdir(folder):
            os.makedirs(folder)

        optimizer = optim.Adam(self.parameters())

        pi_losses_arr = []
        v_losses_arr = []
        for epoch in range(self.epochs):
            self.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.batch_size)

            t = tqdm(range(batch_count), desc=f'Epoch {epoch+1}/{self.epochs}: Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = self.get_nn_input(np.array(boards).astype(np.int32))

                boards = torch.FloatTensor(np.array(boards).astype(np.int32))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.use_cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            pi_losses_arr.append(pi_losses.avg)
            v_losses_arr.append(v_losses.avg)

        f = open(f'{folder}/{filename}', 'a')
        with f:
            writer = csv.writer(f, delimiter=';')
            for l in range(len(pi_losses_arr)):
                writer.writerow([self.id, pi_losses_arr[l], v_losses_arr[l]])
            writer.writerow(['END'])


    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def get_nn_input(self, c_board):
        rigid_layer = self.binary_filter(c_board, 0)
        wood_layer = self.binary_filter(c_board, 1)
        item_layer = self.binary_filter(c_board, 2)
        me_layer = self.binary_filter(c_board, 3)
        enemy_layer = self.binary_filter(c_board, 4)
        blife_layer = self.binary_filter(c_board, 5, 4)
        bstrength_layer = self.binary_filter(c_board, 9, 3)
        flames_layer = self.binary_filter(c_board, 12, 2)
        if len(c_board.shape) == 3:
            stack = np.stack((rigid_layer, wood_layer, item_layer, me_layer, enemy_layer, blife_layer, bstrength_layer, flames_layer), axis=1)
        else:
            stack = np.stack((rigid_layer, wood_layer, item_layer, me_layer, enemy_layer, blife_layer, bstrength_layer, flames_layer), axis=0)
        return stack

    def binary_filter(self, layer, pos, l=1):
        n = ((2 ** l) - 1) << pos
        mask = np.ones(layer.shape, dtype=np.int32) * n
        ml = np.bitwise_and(layer, mask)
        return np.right_shift(ml, pos)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            #print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        #else:
            #print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if self.use_cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint['state_dict'])


class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count