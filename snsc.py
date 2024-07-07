import pickle
from tqdm import tqdm
from collections import Counter

from lpclient import LPClient, Method
from test import init_ns

# generate SNS-C data
def merge_len(most_i, most_t):
    set_i = set([k for (k, _) in most_i])
    set_t = set([k for (k, _) in most_t])
    s = set_i | set_t
    return len(s)

def select_most(c, most):
    selected, counts = [], 0
    for k, v in most:
        selected.append(k)
        counts += v
    all_nodes = sum(c.values())
    print(f'{counts}/{all_nodes}, ratio: {counts/all_nodes}')
    return selected

def run_sns_t_on_train_set(toponame, num_cnodes, lp_kind, obj_kind, data_dir='./data', model_dir='./model', seed=1024, tl=None, ratio=0.7):
    num_agents = 20
    method = Method(cd_method="fan", num_inodes=5,
                    num_tnodes=num_cnodes, lp_kind=lp_kind, obj_kind=obj_kind)
    client = LPClient(toponame=toponame, num_agents=num_agents, data_dir=data_dir,
                      id=id, seed=seed, tl=tl)

    trainset, _ = client.split_dataset(ratio=ratio)
    ns = init_ns(toponame, method, data_dir, model_dir, alpha=1.3)
    i_results, t_results = [], []
    for tm_idx in tqdm(trainset):
        tm = client.TMs[tm_idx]
        inodes, tnodes, extra = ns.select_nodes(method, tm)
        i_results.extend(inodes['Network'])
        t_results.extend(tnodes)
    ci = Counter(i_results)
    ct = Counter(t_results)

    num_inodes = 5
    most_i = ci.most_common(num_inodes)

    if num_cnodes > 10:
        num_cnodes -= num_inodes
    most_t = ct.most_common(num_cnodes)
    
    print('inodes:')
    selected_inodes = select_most(ci, most_i)
    print('tnodes:')
    selected_tnodes = select_most(ct, most_t)
    return selected_inodes, selected_tnodes


def gen_snsc_data(toponame, num_cnodes, data_dir, obj_kind, lp_kind="LP"):
    num_tnodes = num_cnodes
    selected_inodes, selected_tnodes = run_sns_t_on_train_set(toponame, num_cnodes, lp_kind, obj_kind, data_dir)
    result = {'i': selected_inodes, 't': selected_tnodes}
    num_inodes = 5
    if num_cnodes > 10:
        num_cnodes -= num_inodes
    pickle.dump(result, open(
        f"model/snsc-{toponame}-{num_tnodes}-{lp_kind}-{obj_kind}.pkl", "wb"))


def main():
    TNODES = {'GEANT': 4, 'germany50': 10, 'rf1755': 43, 'rf6461': 69}
    for toponame, num_cnodes in TNODES.items():
        gen_snsc_data(toponame=toponame, num_cnodes=num_cnodes,
                      data_dir='./data', obj_kind="MLU")
    for toponame, num_cnodes in TNODES.items():
        gen_snsc_data(toponame=toponame, num_cnodes=num_cnodes,
                      data_dir='./data.mt', obj_kind="MT")
    
if __name__ == '__main__':
    main()
