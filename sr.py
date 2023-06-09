import math

import networkx as nx
import pickle
import numpy as np


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
            filename = f'{self.data_dir}/{self.name}TM'
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


if __name__ == '__main__':
    topo = Topology(name="GEANT")
    tms = Traffic(name="GEANT").load_pickle()
