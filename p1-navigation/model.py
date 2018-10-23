import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """State-action function approximator."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # my code
        self.fc1 = nn.Linear(state_size, 64)
        #self.fc1.weight.data.normal_(0,0.1) # initialization
        self.fc2 = nn.Linear(64, 64)
        #self.fc2.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(64, action_size)
        #self.out.weight.data.normal_(0,0.1) # initialization


    def forward(self, state):
        """Build a network that maps state -> action values."""
        state = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state)
        state = F.relu(state)
        action_value = self.out(state)

        return action_value