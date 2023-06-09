from lpclient import LPClient, Method
from fire import Fire
from copy import deepcopy
import time
import pickle
import numpy as np

from ns import NodeSelector


def init_ns(toponame: str, method: Method, data_dir: str, model_dir: str, alpha: float) -> NodeSelector:
    ns = NodeSelector(toponame, model_dir=model_dir, data_dir=data_dir, alpha=alpha)
    if method.cd_method in ["snsc"]:
        # fake_method = deepcopy(method)
        # fake_method.cd_method = "fan"
        # ns.drl_init(fake_method)
        # model_name = f'model-{ns.toponame}-{fake_method.to_model_name()}.pkl'
        # ns.drl_load_model(model_name)
        pass
    elif method.cd_method in ["sns", "fan"]:
        ns.drl_init(method)
        model_name = f'model-{ns.toponame}-{method.to_model_name()}.pkl'
        ns.drl_load_model(model_name)
    return ns


def run(client: LPClient, method: Method, ns: NodeSelector):
    # 1. spawn a thread to start server
    client.set_method(method)
    client.start_server(logfile=None)
    time.sleep(1)
    # 2. start client
    _, testset = client.split_dataset(ratio=0.7)
    results = client.collect_results(ns, ns_push_sock=None, tm_idx_set=testset)
    time.sleep(1)
    return results


def test(toponame, cd_method, lp_kind, obj_kind, num_inodes, num_tnodes=None, 
         num_agents=20, id=1, seed=1024, tl=None, save=True, alpha=1.2,
         result_dir="./result/", data_dir="./data/", model_dir='./model/'):
    method = Method(cd_method, num_inodes, num_tnodes, lp_kind, obj_kind)
    client = LPClient(toponame=toponame, num_agents=num_agents, data_dir=data_dir,
                      id=id, seed=seed, tl=tl)
    ns = init_ns(toponame, method, data_dir, model_dir, alpha)
    results = run(client, method, ns)

    # print analysis info
    objs, times, pols = results['obj'], results['time'], results['pol']
    maxr, minr, avgr = np.max(objs), np.min(objs), np.mean(objs)
    avg_time = np.mean(times)
    avg_pol = np.mean(pols)
    print(f'topology: {toponame} method: {method}')
    print(f'max:{maxr} min:{minr} avg:{avgr}', flush=True)
    print(f'average solving time: {avg_time:.3f} s', flush=True)
    print(f'average SR policies: {avg_pol:.3f}', flush=True)

    # save result (if needed)
    filename = f"{toponame}-{method.to_str()}.pkl"
    if save:
        f = open(f'{result_dir}/{filename}', 'wb')
        pickle.dump(results, f)


if __name__ == '__main__':
    Fire(test)
