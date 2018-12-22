"""
DQN models
"""

import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        """DQN from DeepMind's nature paper
        
        Parameters
        ----------
        in_channels : int, optional
            number of input channels, i.e., stacked frames (the default is 4)
        num_actions : int, optional
            number of discrete actions we can take (the default is 18)
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # flatten
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc4(x))
        return self.fc5(x)


class DuelingDQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        """Dueling DQN
        
        Parameters
        ----------
        in_channels : int, optional
            number of input channels, i.e., stacked frames (the default is 4)
        num_actions : int, optional
            number of discrete actions we can take (the default is 18)
        """
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4_adv = nn.Linear(7 * 7 * 64, 512)
        self.fc4_val = nn.Linear(7 * 7 * 64, 512)
        self.fc5_adv = nn.Linear(512, num_actions)
        self.fc5_val = nn.Linear(512, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # flatten
        x = x.view(x.size(0), -1)

        # advantage stream. Produces (num_actions, 1)
        adv = F.relu(self.fc4_adv(x))
        adv = F.relu(self.fc5_adv(adv))

        # value stream. Produces (1, 1)
        val = F.relu(self.fc4_val(x))
        val = F.relu(self.fc5_val(val))

        # aggregation
        q = val + adv - adv.mean(1, keepdim=True)
        return q
