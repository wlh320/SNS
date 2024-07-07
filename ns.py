"""node selection"""
import os
import pickle
import random
from functools import lru_cache
from collections import defaultdict
import networkx as nx
import networkx.algorithms.centrality as centrality
import networkit as nk
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from copy import deepcopy

from tqdm import tqdm
from sr import Topology
from drl import PPOGCN, PPOGCN_FAN
from scaleTM import TESolver, Traffic
import fire


# generate str method data

def stretch(lengths, i, l, j):
    eps = 1e-9
    pil = lengths[i][l]
    plj = lengths[l][j]
    pij = lengths[i][j]
    return (pil + plj) / (pij + eps)


def stretch_per_node(toponame, G, k: int, alpha: float) -> list:
    if os.path.exists(f"model/stretch-{toponame}-{k}-{alpha}.pkl"):
        selected = pickle.load(
            open(f"model/stretch-{toponame}-{k}-{alpha}.pkl", "rb"))
        return selected
    """select k nodes for each flow"""
    nodes = [x for x in G.nodes]
    selected = []
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    for i in range(len(nodes)):
        lengths[0][0] = 0

    for i in tqdm(range(len(nodes))):
        for j in range(len(nodes)):
            if i == j:
                continue
            # stretch bounded
            bounded_nodes = sorted(
                [l for l in nodes if stretch(lengths, i, l, j) <= alpha])
            if len(bounded_nodes) >= k:
                nodes4f = np.random.permutation(bounded_nodes)[:k].tolist()
            else:
                nodes4f = np.random.permutation(nodes)[:k].tolist()
            selected.append(nodes4f)
    return selected


def gen_stretch_data(toponame, data_dir, k=5, alpha=1.3):
    G = Topology(toponame, data_dir=data_dir).load()
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
        self.GG = topo.load_networkit(num_nodes=len(self.G.nodes))
        self.solver = TESolver(topo)
        self.num_nodes = len(self.G.nodes)
        self.model_dir = model_dir
        self.data_dir = data_dir
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

    def gsp_centrality(self, k: int = 1) -> list:

        if os.path.exists(f"model/gsp-{self.toponame}-{k}.pkl"):
            selected = pickle.load(
                open(f"model/gsp-{self.toponame}-{k}.pkl", "rb"))
        else:
            selector = nk.centrality.ApproxGroupBetweenness(self.GG, groupSize=k, epsilon=0.005).run()
            selected = selector.groupMaxBetweenness()
            pickle.dump(selected, open(
                f"model/gsp-{self.toponame}-{k}.pkl", "wb"))
        return sorted(selected)

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

    def stretch(self, k: int = 1, alpha: float = 1.3):
        """for each flow"""
        selected = stretch_per_node(self.toponame, self.G, k, alpha=alpha)
        return selected

    def mll_peak(self, k) -> list:
        if os.path.exists(f"{self.model_dir}/mll-{self.toponame}-{k}.pkl"):
            selected = pickle.load(
                open(f"{self.model_dir}/mll-{self.toponame}-{k}.pkl", "rb"))
            return selected
        TMs = Traffic(name=self.toponame, data_dir=self.data_dir).load_pickle()
        peak_tm_idx, peak_vol = 0, 0
        for i, tm in enumerate(TMs):
            vol = sum([sum(line) for line in tm])
            if vol > peak_vol:
                peak_tm_idx, peak_vol = i, vol
        peak_tm = TMs[peak_tm_idx]
        selected = self.mll(tm, k)

        # save file
        file = open(f"{self.model_dir}/mll-{self.toponame}-{k}.pkl", "wb")
        pickle.dump(selected, file)

        return selected
        
    
    def mll(self, tm, k) -> list:
        """most loaded link"""
        paths = dict(nx.all_pairs_shortest_path(self.G))
        utils = defaultdict(float)
        for i in self.G.nodes:
            # compute utlization
            for j in self.G.nodes:
                if i == j: continue
                path = paths[i][j]
                for s, t in zip(path, path[1:]):
                    cap = self.G.edges[(s, t)]['cap']
                    utils[(s, t)] += tm[i][j] / cap
        # most loaded link 
        outgoing_util = defaultdict(float)
        for (src, dst, _) in self.G.edges.data():
            util = utils[(src, dst)]
            outgoing_util[src] = max(outgoing_util[src], util)
        nodes = list(self.G.nodes)
        nodes.sort(key=lambda x: outgoing_util[x], reverse=True)
        return nodes[:k]


    def mll_ecmp(self, tm, k) -> list:
        """most loaded link"""
        # paths = dict(nx.all_pairs_shortest_path(self.G))
        node_scores = {}
        for i in self.G.nodes:
            outgoing_util = defaultdict(float)
            # compute outgoing link load
            for j in self.G.nodes:
                if i == j: continue
                fracs = self.solver.compute_ecmp_link_frac(src=i, dst=j, load=tm[i][j])
                for (s, t), l in fracs.items():
                    if s == i:
                        outgoing_util[t] += l
            # compute outgoing link utlization
            for next_node in outgoing_util.keys():
                cap = self.G.edges[i, next_node]['cap']
                outgoing_util[next_node] /= cap
            # most loaded link 
            node_scores[i] = sorted(outgoing_util.values(), key=lambda v: -v)[0]
        sorted_nodes = [k for k, _ in sorted(
            node_scores.items(), key=lambda x:-x[1])]
        return sorted_nodes[:k]


    def drl_init(self, method):
        assert method.cd_method in ["sns", "fan", "sm"]
        if method.cd_method == "sns" or method.cd_method == "sm":
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
        elif method.cd_method == "gsp":
            inodes = self.gsp_centrality(k=method.num_inodes)
            inodes = {'Network': inodes}
        elif method.cd_method == "deg":
            inodes = self.degree_centrality(k=method.num_inodes)
            inodes = {'Network': inodes}
        elif method.cd_method == "mll":
            inodes = self.mll(tm=tm, k=method.num_inodes)
            # inodes = self.mll_peak(k=method.num_inodes)
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

    def gen_failed_links(self, id: int, num_linkfail: int, known: bool, seed: int):
        """randomly remove some links"""
        if num_linkfail <= 0:
            # do nothing
            return

        ok = False
        while not ok:
            # select links
            idxes = list(range(self.G.number_of_edges()))
            idxes = random.sample(idxes, k=num_linkfail)
            idxes.sort()
            p = 0
            result = []
            for i, (src, dst, _) in enumerate(self.G.edges.data()):
                if p >= len(idxes):
                    break
                if i == idxes[p]:
                    result.append((src, dst))
                    p += 1
            assert num_linkfail == len(result)
            # ensure G is connected
            tmpG = deepcopy(self.G)
            for (src, dst) in result:
                tmpG.remove_edge(src, dst)
            if not nx.is_strongly_connected(tmpG):
                continue
            # found one
            ok = True
            print(f'link {idxes} failed')
            # if known to self, remove in self.G too
            if known:
                for (src, dst) in result:
                    self.G.remove_edge(src, dst)

        return result


def main(toponame="GEANT", num_nc=5, num_cd=5, nc_method="tfidf"):
    ns = NodeSelector(toponame=toponame)
    ns.ncns_and_save(num_nc=num_nc, num_cd=num_cd, nc_method=nc_method)


# test
if __name__ == '__main__':
    # fire.Fire(main)
    pass
