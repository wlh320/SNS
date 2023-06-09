"""node selection"""
import os
import pickle
from functools import lru_cache
import networkx as nx
import networkx.algorithms.centrality as centrality
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

from tqdm import tqdm
from sr import Topology
from drl import PPOGCN, PPOGCN_FAN
import fire


# generate str method data

@lru_cache
def shortest_path_weight(G: nx.Graph, i, j, weight):
    """sum of weight of shortest path"""
    # return nx.shortest_path_length(G, i, j, weight=weight)
    path = nx.shortest_path(G, i, j, weight=weight)
    return nx.path_weight(G, path, weight=weight)


@lru_cache
def stretch(G, i, l, j):
    eps = 1e-9
    pil = shortest_path_weight(G, i, l, weight='weight')
    plj = shortest_path_weight(G, l, j, weight='weight')
    pij = shortest_path_weight(G, i, j, weight='weight')
    return (pil + plj) / (pij + eps)


def stretch_per_node(toponame, G, k: int, alpha: float) -> list:
    if os.path.exists(f"model/stretch-{toponame}-{k}-{alpha}.pkl"):
        selected = pickle.load(
            open(f"model/stretch-{toponame}-{k}-{alpha}.pkl", "rb"))
        return selected
    """select k nodes for each flow"""
    nodes = [x for x in G.nodes]
    selected = []
    for i in tqdm(range(len(nodes))):
        for j in range(len(nodes)):
            if i == j:
                continue
            # stretch bounded
            bounded_nodes = sorted(
                [l for l in nodes if stretch(G, i, l, j) <= alpha])
            if len(bounded_nodes) >= k:
                nodes4f = np.random.permutation(bounded_nodes)[:k].tolist()
            else:
                nodes4f = np.random.permutation(nodes)[:k].tolist()
            selected.append(nodes4f)
    return selected


def gen_stretch_data(toponame, k=5, alpha=1.2):
    G = Topology(toponame).load()
    selected = stretch_per_node(toponame, G, k=k, alpha=alpha)
    pickle.dump(selected, open(
        f"model/stretch-{toponame}-{k}-{alpha}.pkl", "wb"))


class NodeSelector(object):
    """
    Select nodes as middle points
    """

    def __init__(self, toponame: str, model_dir="./model/", data_dir="./data/", alpha=1.3):
        self.toponame = toponame
        topo = Topology(name=toponame, data_dir=data_dir)
        self.G = topo.load()
        self.num_nodes = len(self.G.nodes)
        self.model_dir = model_dir
        self.agent = None
        self.alpha = alpha

    def random_network(self, k: int = 1) -> list:
        """for entire network"""
        nodes = [x for x in self.G.nodes]
        return np.random.permutation(nodes)[:k].tolist()

    def random_flow(self, num_flow, k: int = 1) -> list:
        """for each flow"""
        nodes = [x for x in self.G.nodes]
        selected = []
        for _ in range(num_flow):
            flow_cand = np.random.permutation(nodes)[:k].tolist()
            selected.append(flow_cand)
        return selected

    def sp_centrality(self, k: int = 1) -> list:
        """for entire network"""
        nodes = centrality.betweenness_centrality(self.G)
        sorted_nodes = [k for k, _ in sorted(
            nodes.items(), key=lambda x:-x[1])]
        return sorted_nodes[:k]

    def degree_centrality(self, k: int = 1) -> list:
        """for entire network"""
        nodes = centrality.degree_centrality(self.G)
        sorted_nodes = [k for k, _ in sorted(
            nodes.items(), key=lambda x:-x[1])]
        return sorted_nodes[:k]

    def weighted_sp_centrality(self, k: int = 1) -> list:
        """for entire network"""
        nodes = centrality.betweenness_centrality(self.G, weight='weight')
        sorted_nodes = [k for k, _ in sorted(
            nodes.items(), key=lambda x:-x[1])]
        return sorted_nodes[:k]

    def stretch(self, k: int = 1, alpha: float = 1.2):
        """for each flow"""
        selected = stretch_per_node(self.toponame, self.G, k, alpha=alpha)
        return selected

    def drl_init(self, method):
        assert method.cd_method in ["sns", "fan"]
        if method.cd_method == "sns":
            self.agent = PPOGCN(state_dim=self.num_nodes,
                                action_dim=self.num_nodes, device="cpu")
        elif method.cd_method == "fan":
            self.agent = PPOGCN_FAN(state_dim=self.num_nodes, action_dim=self.num_nodes,
                                    num_inodes=method.num_inodes, num_tnodes=method.num_tnodes,
                                    device="cpu")

    def drl_load_model(self, model_name: str):
        assert self.agent is not None
        self.agent.load_parameters(f"{self.model_dir}/{model_name}")

    def get_edge_index(self) -> torch.Tensor:
        edge_index = list(self.G.edges())
        edge_index = torch.tensor(
            edge_index, dtype=torch.long).t().contiguous()
        return edge_index

    def sns(self, tm, k) -> list:
        s_batch = []
        edge_index = self.get_edge_index()

        state = np.array(tm)
        state = (state - state.mean()) / (state.std() + 1e-12)  # normalization

        state = torch.tensor(state, dtype=torch.float)
        state = Data(x=state, edge_index=edge_index)
        s_batch.append(state)

        action, one_hot_action, log_prob = self.agent.select_action_one_by_one(
            s_batch, k)
        log_prob = log_prob.item()
        extra = {
            's': state,  # Data
            'a': one_hot_action,  # list
            'lp': log_prob,  # float
        }
        return action.tolist(), extra

    def fan(self, tm, mode) -> list:
        assert mode in ["node", "flow"]
        s_batch = []
        edge_index = self.get_edge_index()

        state = np.array(tm)
        state = (state - state.mean()) / (state.std() + 1e-12)  # normalization

        state = torch.tensor(state, dtype=torch.float)
        state = Data(x=state, edge_index=edge_index)
        s_batch.append(state)

        action, one_hot_action, log_prob = self.agent.select_action_one_by_one(
            s_batch, mode)
        log_prob = log_prob.item()
        a_key = 'a' if mode == 'flow' else 'a2' if mode == 'node' else None
        lp_key = 'lp' if mode == 'flow' else 'lp2' if mode == 'node' else None
        extra = {
            's': state,
            a_key: one_hot_action,
            lp_key: log_prob,
        }
        return action.tolist(), extra

    def snsc(self, method, mode) -> list:
        selected = pickle.load(
            open(f"model/snsc-{self.toponame}-{method.num_tnodes}-{method.lp_kind}-{method.obj_kind}.pkl", "rb"))
        assert mode in ['flow', 'node']
        if mode == 'flow':
            return selected['t']
        elif mode == 'node':
            return selected['i']
        return selected

    def select_nodes(self, method, tm):
        # return selected nodes(I & T) and extra information needed for training
        inodes, tnodes, extra = None, None, None
        if method.cd_method == "opt":
            inodes = {'Network': inodes}
        elif method.cd_method == "rand":
            inodes = self.random_network(k=method.num_inodes)
            inodes = {'Network': inodes}
        elif method.cd_method == "sp":
            inodes = self.sp_centrality(k=method.num_inodes)
            inodes = {'Network': inodes}
        elif method.cd_method == "deg":
            inodes = self.degree_centrality(k=method.num_inodes)
            inodes = {'Network': inodes}
        elif method.cd_method == "str":
            inodes = self.stretch(k=method.num_inodes, alpha=self.alpha)
            inodes = {'Flow': inodes}
        elif method.cd_method == "sns":
            inodes, extra = self.sns(tm=tm, k=method.num_inodes)
            inodes = {'Network': inodes}
        elif method.cd_method == "fan":
            if method.num_inodes:
                inodes, i_extra = self.fan(tm=tm, mode="node")
            else:
                inodes, i_extra = None, { 's': None, 'a2': None, 'lp2': None }
            inodes = {'Network': inodes}
            tnodes, t_extra = self.fan(tm=tm, mode="flow")
            i_extra.update(t_extra)
            extra = i_extra
        elif method.cd_method == "snsc":
            inodes = self.snsc(method=method, mode="node")
            inodes = {'Network': inodes}
            tnodes = self.snsc(method=method, mode="flow")
        else:
            pass
        return inodes, tnodes, extra


def main(toponame="GEANT", num_nc=5, num_cd=5, nc_method="tfidf"):
    ns = NodeSelector(toponame=toponame)
    ns.ncns_and_save(num_nc=num_nc, num_cd=num_cd, nc_method=nc_method)


# test
if __name__ == '__main__':
    pass
