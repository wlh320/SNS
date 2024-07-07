import pickle
import tmgen
import numpy as np
import time
import random
from scaleTM import Topology, Traffic, TESolver

num_epochs = 605

INFO = {
    "GEANT": (22, 2016),
    "germany50": (50, 288),
    "rf1755": (87, 288),
    "rf6461": (138, 288),
}


def gen_raw(toponame, out_dir):
    num_nodes, num_epochs = INFO[toponame]
    TMs = []
    for i in range(num_epochs):
        # tm = tmgen.models.random_gravity_tm(num_nodes=num_nodes, mean_traffic=10000000.0)
        tm = tmgen.models.uniform_tm(
            num_nodes=num_nodes, low=0.0, high=10000000.0, num_epochs=num_epochs
        )
        TMs.append(tm.at_time(i))

    path = f"{out_dir}/{toponame}TM.pkl"
    pickle.dump(TMs, open(path, "wb"))
    time.sleep(1)


def gen_outlier(toponame, in_dir, out_dir, ratio, lo, hi):
    assert 0 <= ratio <= 1
    assert 0 <= lo <= 1
    assert hi >= 1

    num_nodes, num_epochs = INFO[toponame]
    num_flows = num_nodes * num_nodes
    num_outliers = int(num_flows * ratio)
    random.seed(1024)
    outliers = random.sample(list(range(num_flows)), k=num_outliers)
    outliers = set(outliers)

    TMs = Traffic(toponame, in_dir).load_pickle()
    new_TMs = []

    for i in range(num_epochs):
        tm = TMs[i]

        for s in range(num_nodes):
            for t in range(num_nodes):
                idx = s * num_nodes + t
                if idx in outliers:
                    times = random.uniform(lo, hi)
                    tm[s][t] = tm[s][t] * times
        new_TMs.append(tm)

    path = f"{out_dir}/{toponame}TM.pkl"
    pickle.dump(new_TMs, open(path, "wb"))
    time.sleep(1)

def scale_TM(dir, toponame, out_dir, load):
    topo = Topology(toponame, dir)
    TMs = Traffic(toponame, dir).load_pickle()
    solver = TESolver(topo)
    solver.precompute_f()
    new_TMs = solver.scale_TMs(TMs, load)

    # dump
    new_TMs = new_TMs.tolist()
    path = f"{out_dir}/{toponame}TM.pkl"
    pickle.dump(new_TMs, open(path, "wb"))


if __name__ == "__main__":
    dir = "./data/"
    for toponame in INFO.keys():
        for burst in [0.05]:
            out_dir = f"./data.burst{burst}/"
            gen_outlier(toponame, in_dir=dir, out_dir=out_dir, ratio=burst, lo=0, hi=3)
