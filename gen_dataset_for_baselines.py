import os
import pathlib
import pickle
import networkx as nx
import numpy as np


class DatasetGenerator:
    def __init__(self, toponame: str, ex_base_dir: str, im_dir: str):
        self.ex_base_dir = ex_base_dir
        self.toponame = toponame
        self.traffic_profile = "gravity_1"
        self.ex_dataset_dir = os.path.join(self.ex_base_dir, self.toponame, self.traffic_profile)
        self.im_dir = im_dir

    def load_my_file(self):
        graph_file = os.path.join(self.im_dir, "data", self.toponame)
        graph = nx.DiGraph()
        with open(graph_file, "r") as f:
            for line in f.readlines():
                line = line.split()[:4]
                src, dst, weight, cap = list(map(int, line))
                graph.add_edge(src, dst, weight=1, bandwidth=str(cap))

        tm_file = os.path.join(self.im_dir, "data", f"{self.toponame}TM.pkl")
        TMs = pickle.load(open(tm_file, "rb"))
        return graph, TMs

    def export_graph_attr(self, graph: nx.DiGraph):
        nx_file = os.path.join(self.ex_base_dir, self.toponame, "graph_attr.txt")
        nx.write_gml(graph, nx_file)

        cap_file = os.path.join(self.ex_dataset_dir, "capacities", "graph.txt")
        pathlib.Path(cap_file).parents[0].mkdir(parents=True, exist_ok=True)

        f = open(cap_file, 'w')
        f.write(f"NODES {graph.number_of_nodes()}\n")
        f.write("label x y\n")
        for i in range(graph.number_of_nodes()):
            f.write(f"N{i} 0.0 0.0\n")
        f.write("\n")

        f.write(f"EDGES {graph.number_of_edges()}\n")
        f.write("label src dest weight bw delay\n")
        i = 0
        for src, dst, attr in graph.edges(data=True):
            weight = attr["weight"]
            bw = attr["bandwidth"]
            f.write(f"Link_{i} {src} {dst} {weight} {bw} 1\n")
            i += 1
        f.close()

    def split_dataset(self, TMs, seed=1024, ratio=0.7):
        """return idxes of splitted data set"""
        num_tm = len(TMs)
        idxes = np.arange(num_tm)
        np_state = np.random.RandomState(seed)
        np_state.shuffle(idxes)

        len_idxes = len(idxes)
        trainsize = int(ratio*len_idxes)
        trainset, testset = idxes[:trainsize], idxes[trainsize:]
        return trainset, testset

    def export_traffic_matrix(self, TMs):
        _, testset = self.split_dataset(TMs, seed=1024, ratio=0.7)

        tm_path = os.path.join(self.ex_dataset_dir, "TM")
        pathlib.Path(tm_path).mkdir(parents=True, exist_ok=True)

        for i, tmidx in enumerate(testset):
            tm = TMs[tmidx]
            n = len(tm)
            tm_file = os.path.join(tm_path, f"TM-{i}")
            f = open(tm_file, 'w')
            strs = []
            cnt = 0
            for s in range(n):
                for t in range(n):
                    if s != t and abs(tm[s][t]) > 1e-9:
                        strs.append(f"demand_{cnt} {s} {t} {tm[s][t]:.5f}\n")
                        cnt += 1
            f.write(f"DEMANDS {cnt}\n")
            f.write(f"label src dest bw\n")
            for s in strs:
                f.write(s)
            f.close()

    def generate(self):
        graph, TMs = self.load_my_file()

        self.export_graph_attr(graph)
        self.export_traffic_matrix(TMs)


if __name__ == "__main__":
    # for toponame in ["GEANT", "germany50", "rf1755", "rf6461"]:
    for toponame in ["nobel"]:
        dg = DatasetGenerator(
            toponame=toponame,
            ex_base_dir="/home/wlh/repos/tnsm2023-deepls/MARL-GNN-TE/datasets/",
            im_dir="./",
        )
        dg.generate()
