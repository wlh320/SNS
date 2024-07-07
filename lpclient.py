import pickle
import zmq
import logging
import numpy as np
from tqdm import tqdm
import subprocess
import time

from sr import Traffic, Topology
from ns import NodeSelector

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)

CD_METHODS = set(['opt', 'rand', 'sp', 'gsp', 'deg', 'str', 'sns', 'fan', 'snsc', 'mll'])


class Method:
    def __init__(self, cd_method, num_inodes, num_tnodes, lp_kind, obj_kind):
        assert lp_kind in ['LP', 'ILP']
        assert obj_kind in ['MLU', 'MT']
        assert cd_method in CD_METHODS
        if cd_method in ["fan", "snsc"]:
            assert num_tnodes is not None
        else:
            assert num_tnodes is None
        self.cd_method = cd_method
        self.num_inodes = num_inodes  # int | None
        self.num_tnodes = num_tnodes  # int | None
        self.lp_kind: str = lp_kind  # LP | ILP
        self.obj_kind: str = obj_kind  # MLU | MT

    def __repr__(self) -> str:
        return f"[method: {self.cd_method}-i{self.num_inodes}-t{self.num_tnodes} lp: {self.lp_kind} obj: {self.obj_kind} ]"

    def to_str(self):
        return f"{self.cd_method}-i{self.num_inodes}-t{self.num_tnodes}-{self.lp_kind}-{self.obj_kind}"

    def to_model_name(self):
        return f"{self.cd_method}-i{self.num_inodes}-t{self.num_tnodes}-{self.lp_kind}-{self.obj_kind}"


class LPClient:
    """
    Communicate with server written in rust.
    Send Tasks, and get Solution.
    Data packed with pickle, read/write with zmq ipc

        Task structure:
        {
            "idx": Some(1) | None,
            "lp_kind": "LP" | "ILP",
            "obj_kind": "MLU" | "MT",
            "cands": {"Network" | "Node" | "Flow": list } | None,
            "num_cd": int | None,
            "tnodes": list[int] | None,
        }
        Result strucure
        idx, obj, time
    """

    def __init__(self, toponame: str, num_agents: int, id=1, data_dir="./data/",
                 seed=1024, tl=None, num_linkfail=0):
        self.id = id
        self.toponame = toponame
        self.data_dir = data_dir
        self.num_agents = num_agents
        self.seed = seed

        self.TMs = Traffic(name=self.toponame, data_dir=data_dir).load_pickle()
        self.push_sock = None  # send task to lpserver
        self.pull_sock = None  # recv result from lpserver
        self.tl = tl
        self.num_linkfail = num_linkfail

        self.method = None

        logger.info(
            f'Run #{self.id} info: topo {self.toponame} with {self.num_agents} agents')
    
    def __repr__(self) -> str:
        s = f"""lpclient [topo: {self.toponame}, data_dir: {self.data_dir},
                    num_agents: {self.num_agents} seed: {self.seed}, time_limit: {self.tl}
                ]"""
        return s

    def set_method(self, method: Method):
        self.method = method
        logger.info(f'Set method: {self.method}')

    def split_dataset(self, ratio=0.7):
        """return idxes of splitted data set"""
        num_tm = len(self.TMs)
        idxes = np.arange(num_tm)
        np_state = np.random.RandomState(self.seed)
        np_state.shuffle(idxes)

        len_idxes = len(idxes)
        trainsize = int(ratio*len_idxes)
        trainset, testset = idxes[:trainsize], idxes[trainsize:]
        return trainset, testset

    def send_task(self, ns: NodeSelector, ns_push_sock, tm_idx):
        t = time.time()
        tm = self.TMs[tm_idx]
        inodes, tnodes, extra_info = ns.select_nodes(method=self.method, tm=tm)
        # TODO: send data needed by training to zmq socket
        ns_time = time.time() - t
        task = {
            'idx': int(tm_idx),
            'lp_kind': self.method.lp_kind,
            'obj_kind': self.method.obj_kind,
            'cands': inodes,  # cand_levels: 'Network' | 'Node' | 'Flow'
            'num_cd': self.method.num_inodes,
            'tnodes': tnodes
        }
        msg = pickle.dumps(task)
        self.push_sock.send(msg)
        # send extra data for training
        if ns_push_sock:
            extra = {'idx': int(tm_idx)}
            extra.update(extra_info)
            ns_push_sock.send_pyobj(extra)
        return task, ns_time

    def parse_result(self, result):
        ans = pickle.loads(result)
        idx, obj, time, pol = ans['idx'], ans['obj'], ans['time'], ans['pol']
        return idx, obj, time, pol

    def send_quit_signal(self):
        data = {
            'idx': None,
            'lp_kind': self.method.lp_kind,
            'obj_kind': self.method.obj_kind,
            # cand_levels: 'Network' | 'Node' | 'Flow'
            'cands': {'Network': None},
            'num_cd': self.method.num_inodes,
            'tnodes': None
        }
        self.push_sock.send(pickle.dumps(data))

    def collect_results(self, ns, ns_push_sock, tm_idx_set, quit=True):
        """
        select nodes using `ns` for each tm in `tm_idx_set` as tasks,  
        send tasks to server, 
        recv results.
        [optional: send quit signal to server if `quit`]
        """
        c = zmq.Context()
        self.push_sock = c.socket(zmq.PUSH)
        self.push_sock.bind(f"ipc:///tmp/tasks{self.id}")
        self.pull_sock = c.socket(zmq.PULL)
        self.pull_sock.bind(f"ipc:///tmp/results{self.id}")
        time_dict = dict()

        curr = 0  # counter of task

        # push initial tasks
        while curr < min(self.num_agents, len(tm_idx_set)):
            idx = tm_idx_set[curr]
            _, ns_time = self.send_task(ns, ns_push_sock, idx)
            time_dict[idx] = ns_time
            curr += 1

        # pull results
        ids, objs, times, pols = [], [], [], []
        bar = tqdm(total=len(tm_idx_set), leave=False, ascii=True)
        while True:
            result = self.pull_sock.recv()
            if result == b"quit":
                break

            # handle result
            id, obj, time, pol = self.parse_result(result)
            ns_time = time_dict[id]
            # print(f'Recv result: [id: {id} mlu: {mlu} time: {time}]')
            ids.append(id)
            objs.append(obj)
            times.append(time + ns_time)
            pols.append(pol)
            bar.update(1)

            # quit if all task finished, but does not quit server
            # break without send quit signal to server
            if len(ids) == len(tm_idx_set):
                break

            # push more tasks
            if curr < len(tm_idx_set):
                idx = tm_idx_set[curr]
                _, ns_time = self.send_task(ns, ns_push_sock, idx)
                time_dict[idx] = ns_time
                curr += 1
            elif curr == len(tm_idx_set):
                if quit:
                    self.send_quit_signal()
                curr += 1
            else:
                pass
        bar.close()

        results = {
            'id': ids,
            'obj': objs,
            'time': times,
            'pol': pols
        }
        return results

    def start_server(self, logfile=None):
        """start lpserver"""
        binpath = '/home/wlh/coding/SNS2024/lpserver/target/release/lpserver'
        # prepare arguments
        stdout = subprocess.DEVNULL if logfile is None else open(logfile, 'w')
        tl = '' if self.tl is None else f'--time-limit={self.tl}'
        linkfail = '' if self.num_linkfail == 0 else f'--num-linkfail={self.num_linkfail}'
        # start subprocess
        cmd = [f'{binpath}', f'--id={self.id}', f'--data-dir={self.data_dir}',
               f'--toponame={self.toponame}', f'--num-agents={self.num_agents}', tl, linkfail]
        cmd = ' '.join(cmd).strip().split()
        print(f'lpserver: {" ".join(cmd)}')
        subprocess.Popen(cmd, env=None, shell=False,
                         stdout=stdout)
