from lpclient import LPClient, Method
from fire import Fire
from copy import deepcopy
import time
import pickle
import numpy as np
import zmq
import random

from ns import NodeSelector


def init_ns(toponame: str, method: Method, data_dir: str, model_dir: str, alpha: float) -> NodeSelector:
    ns = NodeSelector(toponame, model_dir=model_dir, data_dir=data_dir, alpha=alpha)
    if method.cd_method in ["snsc"]:
        pass
    elif method.cd_method in ["sns", "fan"]:
        ns.drl_init(method)
        model_name = f'model-{ns.toponame}-{method.to_model_name()}.pkl'
        ns.drl_load_model(model_name)
    return ns


def send_failed_links(id, failed_links):
    # send to client
    c = zmq.Context()
    push_sock = c.socket(zmq.PUSH)
    push_sock.bind(f"ipc:///tmp/link_failures{id}")

    msg = pickle.dumps(failed_links)
    push_sock.send(msg)


def run_with_failed_links(id: int, client: LPClient, method: Method, ns: NodeSelector, num_tm_each_run: int, failed_links, seed: int):
    # 1. spawn a thread to start server
    client.set_method(method)
    client.start_server(logfile=None)
    time.sleep(1)
    # 2. generate dataset
    _, testset = client.split_dataset(ratio=0.7)

    np_state = np.random.RandomState(seed)
    testset = np_state.choice(testset, size=num_tm_each_run, replace=False)

    # assert num_tm_each_run == 20
    # testset = testset[:10]
    #
    # 3. send failed links to server
    send_failed_links(id, failed_links)

    # 4. start client
    results = client.collect_results(ns, ns_push_sock=None, tm_idx_set=testset)
    time.sleep(1)
    return results



def test(toponame, cd_method, lp_kind, obj_kind, num_inodes, num_tnodes=None, 
         num_agents=20, id=1, seed=1024, tl=None, save=True, alpha=1.3,
         num_linkfail=0, num_tm_each_run=20, run_times=1, known=True,
         result_dir="./result/", data_dir="./data/", model_dir='./model/'):
    method = Method(cd_method, num_inodes, num_tnodes, lp_kind, obj_kind)
    client = LPClient(toponame=toponame, num_agents=num_agents, data_dir=data_dir,
                      id=id, seed=seed, tl=tl, num_linkfail=num_linkfail)

    # do multiple runs each with 
    results_array = []
    all_avg = 0
    random.seed(seed)
    for i in range(run_times):
        ns = init_ns(toponame, method, data_dir, model_dir, alpha)
        # robust scenerio
        failed_links = ns.gen_failed_links(id=id, num_linkfail=num_linkfail, known=known, seed=seed)

        results = run_with_failed_links(id, client, method, ns, num_tm_each_run, failed_links, seed=seed)

        results_array.append(results)

        # print analysis info
        objs, times, pols = results['obj'], results['time'], results['pol']
        maxr, minr, avgr = np.max(objs), np.min(objs), np.mean(objs)
        avg_time = np.mean(times)
        avg_pol = np.mean(pols)
        print(f'topology: {toponame} method: {method} num_linkfail: {num_linkfail}')
        print(f'max:{maxr} min:{minr} avg:{avgr}', flush=True)
        print(f'average solving time: {avg_time:.3f} s', flush=True)
        print(f'average SR policies: {avg_pol:.3f}', flush=True)
        all_avg += (avgr / run_times)

    print('===================================================================')
    print(f'topology: {toponame} method: {method} num_linkfail: {num_linkfail}')
    print(f'avg result: {all_avg}')
    print('===================================================================')

    # save result (if needed)
    known = "known" if known else "unknown"
    filename = f"{toponame}-{method.to_str()}-linkfail-{num_linkfail}-{known}.pkl"
    if save:
        f = open(f'{result_dir}/{filename}', 'wb')
        pickle.dump(results_array, f)


if __name__ == '__main__':
    Fire(test)
