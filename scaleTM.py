import os
import math
from collections import defaultdict

import networkx as nx
import pickle
import numpy as np
from tqdm import tqdm
import os
from fire import Fire


class Topology(object):
    def __init__(self, name, data_dir='./data/'):
        self.name = name
        self.data_dir = data_dir

    def load(self):
        G = nx.DiGraph()
        try:
            filename = f'{self.data_dir}/{self.name}'
            with open(filename, 'r') as f:
                for line in f.readlines():
                    line = line.split()[:4]
                    src, dst, weight, cap = list(map(int, line))
                    G.add_edge(src, dst, weight=weight, cap=cap)
        except Exception as e:
            print(f'failed to load topology {self.name}')
            print(e)
        return G


class Traffic(object):
    def __init__(self, name, data_dir='./data/'):
        self.name = name
        self.data_dir = data_dir

    def load(self):
        TMs = []
        try:
            filename = f'{self.data_dir}{self.name}'
            with open(filename, 'r') as f:
                for line in f.readlines():
                    line = line.split()
                    tm = np.array(list(map(float, line)))
                    tm = tm.reshape((math.isqrt(len(tm)), -1))
                    TMs.append(tm)
        except Exception as e:
            print(f'failed to load traffic matrices {self.name}')
            print(e)

        #  Unit: (100 bytes / 5 minutes)
        return np.array(TMs) * 100 * 8 / 300 / 1000  # kbps

    def load_pickle(self):
        filename = f'{self.data_dir}/{self.name}TM.pkl'
        TMs = pickle.load(open(filename, 'rb'))
        return TMs


class TESolver(object):

    def __init__(self, topo: Topology):
        self.f_dict = {}
        # self.g_dict = {}
        self.topo = topo
        self.G = self.topo.load()

    def handle_G(self):
        edges, caps = [], []
        for src, dst, attr in self.G.edges.data():
            caps.append(attr['cap'])
            edges.append((src, dst))
        return caps, edges

    def handle_TM(self, TM: np.array):
        flows, demands = [], []
        num_nodes = len(TM)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                if TM[i][j] != 0:
                    flows.append((i, j))
                    demands.append(TM[i][j])
        return flows, demands

    def precompute_f(self):
        num_nodes = len(self.G.nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    ans = self.compute_ecmp_link_frac(i, j, load=1.0)
                    self.f_dict[i, j] = ans

    def compute_ecmp_link_frac(self, src, dst, load=1.0):
        ans = defaultdict(int)
        try:
            paths = list(nx.all_shortest_paths(
                self.G, src, dst, weight='weight'))
            # build DAG
            dag = nx.DiGraph()
            node_succ = defaultdict(set)
            node_load = defaultdict(int)
            for p in paths:
                for s, t in zip(p, p[1:]):
                    node_succ[s].add(t)
                    dag.add_nodes_from([s, t])
                    dag.add_edge(s, t, frac=0.0)
            # compute fraction
            node_load[src] = load
            for node in nx.topological_sort(dag):
                nexthops = node_succ[node]
                if not nexthops:
                    continue
                nextload = node_load[node] / len(nexthops)
                for nexthop in nexthops:
                    dag[node][nexthop]['frac'] += nextload
                    node_load[nexthop] += nextload
            for s, t in dag.edges:
                ans[s, t] = dag[s][t]['frac']

        except (KeyError, nx.NetworkXNoPath):
            print("Error, no path for %s to %s in apply_ecmp_flow()" % (src, dst))
        return ans

    def f(self, i, j, e):
        """ecmp fraction of edges"""
        if (i, j) not in self.f_dict:
            return 0
        return self.f_dict[i, j].get(e, 0)

    def g(self, i, j, k, e):
        return self.f(i, k, e) + self.f(k, j, e)

    def handle(self, TM):
        C, E = self.handle_G()  # Capacity, Edge
        F, D = self.handle_TM(TM)  # Flow, Demand
        return C, E, F, D

    def solve_ecmp(self, TM):
        C, E = self.handle_G()  # Capacity, Edge
        F, D = self.handle_TM(TM)
        num_e, num_f = len(E), len(F)
        U = []
        for e in range(num_e):
            u = 0
            for f in range(num_f):
                u += D[f] * self.f(F[f][0], F[f][1], E[e]) / C[e]
            U.append(u)
        print(f'Max load:{max(U)}')
        return max(U)

    def scale_TMs(self, TMs, load=1.0):
        ans = []
        for TM in tqdm(TMs):
            scaling = self.solve_ecmp(TM)
            newTM = np.array(TM) * (load / scaling)
            ans.append(newTM)
        return np.array(ans)


def scale_TM(dir, toponame, out_dir, load):
    topo = Topology(toponame, dir)
    TMs = Traffic(toponame, dir).load_pickle()
    solver = TESolver(topo)
    solver.precompute_f()
    new_TMs = solver.scale_TMs(TMs, load)

    # dump
    new_TMs = new_TMs.tolist()
    path = f'{out_dir}/{toponame}TM.pkl'
    pickle.dump(new_TMs, open(path, 'wb'))


def main(dir, out_dir, load):
    for topo in ["GEANT", "germany50", "rf1755", "rf6461"]:
        scale_TM(dir, topo, out_dir, load)
    # scale_TM(dir, "Abilene", out_dir, load)
    # scale_TM(dir, "nobel", out_dir, load)


if __name__ == '__main__':
    # # 0.05
    # in_dir = './data.burst0.05/'
    # out_dir = './data.burst0.05.mt/'
    # main(in_dir, out_dir, load=3.0)
    #
    # # 0.1
    # in_dir = './data.burst0.1/'
    # out_dir = './data.burst0.1.mt/'
    # main(in_dir, out_dir, load=3.0)
    #
    # # 0.2
    # in_dir = './data.burst0.2/'
    # out_dir = './data.burst0.2.mt/'
    # main(in_dir, out_dir, load=3.0)
    scale_TM(dir='/home/wlh/repos/TMgen/', toponame="rf1755", out_dir='./', load=1.0)
