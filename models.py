import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv
from torch_geometric.data import Data
from config import config


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class GCN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, action_dim, dropout):
        super(GCN, self).__init__()
        self.state_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.fc_hidden_dim = config.hidden_dim

        self.gc1 = GCNConv(feature_dim, hidden_dim)
        self.dropout = dropout

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.fc_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.fc_hidden_dim, self.action_dim),
        )

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        x = F.leaky_relu(x)
        x = x.reshape((-1, self.state_dim, self.hidden_dim))
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


class GCN_FAN(nn.Module):
    """select both nodes and flows"""

    def __init__(self, feature_dim, hidden_dim, action_dim, dropout):
        super(GCN_FAN, self).__init__()
        self.state_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.fc_hidden_dim = config.hidden_dim
        self.fc1_hidden_dim = config.flow_hidden_dim

        def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(
            x, 0), torch.nn.init.calculate_gain('leaky_relu'))

        def init1_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(
            x, 0))

        self.gc1 = init_(GCNConv(feature_dim, hidden_dim))
        self.dropout = dropout

        # select flow
        self.fc1 = nn.Sequential(
            init_(nn.Linear(self.hidden_dim, self.fc1_hidden_dim)),
            nn.LeakyReLU(),
            init1_(nn.Linear(self.fc1_hidden_dim, self.action_dim)),
        )

        # select node
        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.fc_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.fc_hidden_dim, self.action_dim),
        )

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        x = F.leaky_relu(x)
        x = x.reshape((-1, self.state_dim, self.hidden_dim))
        x = x.mean(dim=1)
        x1, x2 = self.fc1(x), self.fc2(x)
        return x1, x2

    def forward_flow(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        x = F.leaky_relu(x)
        x = x.reshape((-1, self.state_dim, self.hidden_dim))
        x = x.mean(dim=1)
        x = self.fc1(x)
        return x

    def forward_node(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        x = F.leaky_relu(x)
        x = x.reshape((-1, self.state_dim, self.hidden_dim))
        x = x.mean(dim=1)
        x = self.fc2(x)
        return x
