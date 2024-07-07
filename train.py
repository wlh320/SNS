import numpy as np
import time

from fire import Fire
from lpclient import LPClient, Method
from ns import NodeSelector
import zmq

from sr import Traffic
from tqdm import tqdm


class SNSTrainer:
    def __init__(self, toponame: str, method: Method, client: LPClient, batch_size: int,
                 model_dir: str, data_dir: str):
        self.toponame = toponame
        self.method = method
        self.client = client
        self.batch_size = batch_size
        self.baseline = dict()
        self.model_dir = model_dir
        # init ns
        self.ns = NodeSelector(toponame, data_dir=data_dir)
        if method.cd_method in ["sns", "fan"]:
            self.ns.drl_init(method)
        # init client
        client.set_method(method)

    def test_sns_selection(self):
        TMs = Traffic(self.toponame).load_pickle()
        idx = 0
        tm = TMs[idx]
        res = self.ns.sns(tm, k=5)
        print(res)

    def get_reward(self, idx: int, results: dict):
        """
        results = {
            'id': ids, // list
            'obj': objs, // list
            'time': times, // list
        }
        """
        pos = results['id'].index(idx)
        reward = results['obj'][pos]
        if self.method.obj_kind == 'MLU':
            reward = 1.0 / reward
        return reward

    def generate_subset(self, dataset: np.array) -> np.array:
        # TODO: group and choice, or direct choice, are they the same?
        subset = np.random.choice(dataset, size=self.batch_size, replace=True)
        return subset

    def compute_advantage(self, idx: int, reward: float):
        total_reward, cnt = self.baseline.get(idx, (reward, 1))
        avg_reward = total_reward / cnt
        advantage = reward - avg_reward
        # update baseline
        if idx in self.baseline:
            self.baseline[idx] = (total_reward + reward, cnt + 1)
        else:
            self.baseline[idx] = (reward, 1)
        return advantage

    def train(self, id, step, ratio=0.7, suffix=None):
        self.client.start_server(logfile=None)
        time.sleep(1)

        # inproc zmq sockets
        c = zmq.Context()
        ns_push_sock = c.socket(zmq.PUSH)
        ns_push_sock.bind(f"inproc:///tmp/extra{id}")
        ns_pull_sock = c.socket(zmq.PULL)
        ns_pull_sock.connect(f"inproc:///tmp/extra{id}")

        trainset, _ = self.client.split_dataset(ratio=ratio)
        
        bestr = 0

        for i in range(1, step+1):
            print(f"step {i}/{step}")

            # 1. generate subset
            subset = self.generate_subset(trainset)

            # 2. send to lpserver through lpclient
            results = self.client.collect_results(
                self.ns, ns_push_sock, subset, quit=False)

            # 3. recv trainging input
            s_batch, a_batch, r_batch, ad_batch, lp_batch = [], [], [], [], []
            a2_batch, lp2_batch = [], []

            # helper functions
            def append_data_to_batch(s, a, lp, a2, lp2, r, adv):
                s_batch.append(s)
                a_batch.append(a)
                lp_batch.append(lp)
                if self.method.cd_method == "fan":
                    a2_batch.append(a2)
                    lp2_batch.append(lp2)
                r_batch.append(r)
                ad_batch.append(adv)

            def parse_extra_info(extra):
                idx, s, a, lp = extra['idx'], extra['s'], extra['a'], extra['lp']
                if self.method.cd_method == "fan":
                    a2, lp2 = extra['a2'], extra['lp2']
                else:
                    a2, lp2 = None, None
                return idx, s, a, lp, a2, lp2

            while len(s_batch) != len(subset):
                # recv extra info
                extra = ns_pull_sock.recv_pyobj()
                # parse extra info
                idx, s, a, lp, a2, lp2 = parse_extra_info(extra)
                # get reward and adv
                r = self.get_reward(idx, results)
                adv = self.compute_advantage(idx, r)
                # append to batch
                append_data_to_batch(s, a, lp, a2, lp2, r, adv)

            # 4. print result
            objs, times = results['obj'], results['time']
            maxr, minr, avgr = np.max(objs), np.min(objs), np.mean(objs)
            avg_time = np.mean(times)
            print(f'max:{maxr} min:{minr} avg:{avgr}', flush=True)
            print(f'average solving time: {avg_time:.3f} s', flush=True)

            # 5. update ns model
            if self.method.cd_method == "sns":
                self.ns.agent.update(
                    s_batch, a_batch, r_batch, ad_batch, lp_batch)
            elif self.method.cd_method == "fan":
                if self.method.num_inodes:
                    self.ns.agent.update(
                        s_batch, a_batch, a2_batch, r_batch, ad_batch, lp_batch, lp2_batch, mode='MGDA')
                else:
                    self.ns.agent.update(
                        s_batch, a_batch, a2_batch, r_batch, ad_batch, lp_batch, lp2_batch, mode='flow')
            
            avgr = np.mean(r_batch)
            if avgr > bestr:
                # save best model
                filename = f'{self.model_dir}/model-{self.toponame}-{self.method.to_model_name()}.pkl'
                self.ns.agent.save_parameters(filename)
                bestr = avgr
                print(f'current best model: {avgr}')

        # save model
        # filename = f'{self.model_dir}/model-{self.toponame}-{self.method.to_model_name()}.pkl'
        # self.ns.agent.save_parameters(filename)
        print(f'current best model: {bestr}')
        print(f'final step model: {avgr}')

        filename = f'{self.model_dir}/model-{self.toponame}-{self.method.to_model_name()}-finalstep.pkl'
        self.ns.agent.save_parameters(filename)

        # tell server to quit
        self.client.send_quit_signal()


def train(toponame, cd_method, lp_kind, obj_kind, num_inodes, num_tnodes=None,
          batch_size=20 * 3, num_agents=20, id=1, seed=1024, tl=None, step=200,
          model_dir='./model/', data_dir='./data/'):
    method = Method(cd_method, num_inodes, num_tnodes, lp_kind, obj_kind)
    client = LPClient(toponame=toponame, num_agents=num_agents, data_dir=data_dir,
                      id=id, seed=seed, tl=tl)
    trainer = SNSTrainer(toponame, method, client,
                         batch_size, model_dir, data_dir)
    # suffix = f"{toponame}-{method.to_str()}"
    suffix = None
    print(f'method: {method}')
    print(f'lpclient: {client}')
    trainer.train(id=id, step=step, suffix=suffix)


if __name__ == '__main__':
    Fire(train)
