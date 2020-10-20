import os
import torch
import torch.nn as nn

class PommermanNNet(nn.Module):

    def __init__(self):
        super(PommermanNNet, self).__init__()

        self.optim = torch.optim.Adam(self.parameters(), lr=0.00005)

        self.conv1 = initConvLayer(4, 12)
        self.conv2 = initConvLayer(12, 24)
        self.conv3 = initConvLayer(24, 36)
        self.conv4 = initConvLayer(36, 36)
        self.conv5 = initConvLayer(36, 48)

        self.dense = nn.Sequential(
            nn.Linear(720, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    # Applies forward propagation to the inputs
    def forward(self, frameStack):
        out = self.conv1(frameStack)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.dense(out.view(out.size(0), -1))

        return torch.softmax(out, dim=-1)

    def save_current_state(self, folder, filename) -> None:
        state = {
            'actor_critic': self.state_dict(),
            'optim': self.optim.state_dict()
        }
        try:
            torch.save(state, os.path.join(folder, filename))
        except:
            print("Nice little exception...")

    def load_current_state(self, folder, filename) -> None:
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            chkpt = torch.load(path)
            self.actor_critic.load_state_dict(chkpt['actor_critic'])
            self.optim.load_state_dict(chkpt['optim'])

            print("Loaded ", path)
        else:
            print("No checkpoint found")

def initConv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    nn.init.xavier_uniform_(conv.weight)
    return conv

def initConvLayer(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Sequential(
        initConv2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU(),
        nn.MaxPool2d(2, 2))