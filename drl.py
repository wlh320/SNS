import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from config import config
from models import GCN, GCN_FAN
from min_norm_solvers import MinNormSolver


class PPOGCN:
    """single-step version of PPO with GCN"""

    def __init__(self, state_dim, action_dim, device):
        self.lr = config.ppo_lr
        self.entropy_lr = config.ppo_entropy_lr
        self.k_epoch = config.k_epoch
        self.device = device
        self.eps_clip = config.eps_clip
        self.action_dim = action_dim
        self.gcn = GCN(state_dim, hidden_dim=config.hidden_features,
                       action_dim=action_dim, dropout=0.25).to(device)
        self.optim = optim.Adam(self.gcn.parameters(), lr=self.lr)
        self.scheduler = ExponentialLR(self.optim, gamma=0.98)

    def select_action(self, state, k):
        self.gcn.eval()
        loader = DataLoader(state, batch_size=len(state))
        for data in loader:
            data = data.to(self.device)
            logits = self.gcn(data)
            # print(logits.shape)
            logits = logits.reshape((len(state), -1))
            probs = F.softmax(logits, dim=1).cpu()
            action = torch.multinomial(probs, k).squeeze()
            probs = torch.unsqueeze(probs, dim=-1)
            one_hot_action = np.eye(self.action_dim, dtype=np.float32)[
                np.array(action)]
            one_hot_action = torch.Tensor(one_hot_action)
            log_probs = torch.log(torch.squeeze(
                torch.matmul(one_hot_action, probs)+1e-9)).sum(dim=1)
        return action.detach().numpy(), one_hot_action.detach().numpy(), log_probs.detach().numpy()

    def select_action_one_by_one(self, state, k):
        """only for test"""
        self.gcn.eval()
        loader = DataLoader(state, batch_size=1)
        # probs_list = np.array([])
        for data in loader:
            data = data.to(self.device)
            logits = self.gcn(data)
            # print(logits.shape)
            logits = logits.reshape(-1)
            probs = F.softmax(logits, dim=-1).cpu()
            # probs_list = probs.detach().numpy()
            action = torch.multinomial(probs, k)

            one_hot_action = np.eye(self.action_dim, dtype=np.float32)[
                np.array(action)]
            one_hot_action = torch.Tensor(one_hot_action)
            log_probs = torch.log(torch.matmul(
                one_hot_action, probs)+1e-9).sum()
        # print(actions)
        return action.detach().numpy(), one_hot_action.detach().numpy(), log_probs

    def update(self, s_batch, a_batch, r_batch, ad_batch, lp_batch):
        # s_batch = torch.FloatTensor(s_batch).detach().to(self.device) # matrix
        # s_batch = ts_batch.to(self.device)
        loader = DataLoader(s_batch, batch_size=len(s_batch))
        actions = torch.Tensor(a_batch).detach().to(self.device)  # one-hot
        rewards = torch.Tensor(r_batch).detach().to(self.device)
        advantages = torch.FloatTensor(ad_batch).detach().to(self.device)
        old_log_probs = torch.FloatTensor(lp_batch).detach().to(self.device)
        eps = 1e-9

        # reward normalization
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + eps)

        self.gcn.train()
        for data in loader:
            data = data.to(self.device)
            for _ in range(self.k_epoch):
                # print(data)
                logits = self.gcn(data)
                logits = logits.reshape((len(s_batch), -1))
                probs = F.softmax(logits, dim=1)
                # print(logits)
                # print(probs)
                m = Categorical(probs)
                entropy = m.entropy()
                probs = torch.unsqueeze(probs, dim=-1)
                log_probs = torch.log(torch.squeeze(
                    torch.matmul(actions, probs)+eps)).sum(dim=1)
                # advantages = rewards - v_values.detach()

                # surrogate loss
                ratios = torch.exp(log_probs - old_log_probs.detach())
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                    1+self.eps_clip) * advantages
                s_loss = -torch.min(surr1, surr2).mean()
                # entropy loss
                e_loss = self.entropy_lr * entropy.mean()

                total_loss = s_loss - e_loss
                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()

        print(
            f'policy: {s_loss.mean()} entropy: {e_loss.mean()} total: {total_loss} lr: {lr}', flush=True)

    def get_parameters(self):
        return self.gcn.state_dict()

    def set_parameters(self, state_dict):
        self.gcn.load_state_dict(state_dict)

    def save_parameters(self, name):
        torch.save(self.gcn.state_dict(), f'{name}')

    def load_parameters(self, name):
        self.gcn.load_state_dict(
            torch.load(f'{name}', map_location=torch.device('cpu')), strict=False
        )


class PPOGCN_FAN:
    """single-step version of PPO with GCN (flow and node)"""

    def __init__(self, state_dim, action_dim, device, num_inodes, num_tnodes):
        self.lr = config.ppo_lr
        self.entropy_lr = config.ppo_entropy_lr
        self.k_epoch = config.k_epoch
        self.device = device
        self.eps_clip = config.eps_clip
        self.action_dim = action_dim
        self.gcn = GCN_FAN(state_dim, hidden_dim=config.hidden_features,
                           action_dim=action_dim, dropout=0.25).to(device)
        self.optim = optim.Adam(self.gcn.parameters(), lr=self.lr)
        self.scheduler = ExponentialLR(self.optim, gamma=0.98)
        self.num_inodes = num_inodes
        self.num_tnodes = num_tnodes

    def select_action(self, state, mode):
        """mode = flow or node"""
        assert mode == 'flow' or mode == 'node'
        self.gcn.eval()
        loader = DataLoader(state, batch_size=len(state))
        for data in loader:
            data = data.to(self.device)
            if mode == 'flow':
                logits = self.gcn.forward_flow(data)
                k = self.num_tnodes
            elif mode == 'node':
                logits = self.gcn.forward_node(data)
                k = self.num_inodes
            logits = logits.reshape((len(state), -1))
            probs = F.softmax(logits, dim=1).cpu()
            action = torch.multinomial(probs, k).squeeze()
            probs = torch.unsqueeze(probs, dim=-1)
            one_hot_action = np.eye(self.action_dim, dtype=np.float32)[
                np.array(action)]
            one_hot_action = torch.Tensor(one_hot_action)
            log_probs = torch.log(torch.squeeze(
                torch.matmul(one_hot_action, probs)+1e-9)).sum(dim=1)
        return action.detach().numpy(), one_hot_action.detach().numpy(), log_probs.detach().numpy()

    def select_action_one_by_one(self, state, mode):
        """only for test"""
        assert mode == 'flow' or mode == 'node'
        self.gcn.eval()
        # actions = []
        loader = DataLoader(state, batch_size=1)
        # probs_list = np.array([])
        for data in loader:
            data = data.to(self.device)
            if mode == 'flow':
                logits = self.gcn.forward_flow(data)
                k = self.num_tnodes
            elif mode == 'node':
                logits = self.gcn.forward_node(data)
                k = self.num_inodes
            # print(logits.shape)
            logits = logits.reshape(-1)
            probs = F.softmax(logits, dim=-1).cpu()
            # probs_list = probs.detach().numpy()
            action = torch.multinomial(probs, k)

            one_hot_action = np.eye(self.action_dim, dtype=np.float32)[
                np.array(action)]
            one_hot_action = torch.Tensor(one_hot_action)
            log_probs = torch.log(torch.matmul(
                one_hot_action, probs)+1e-9).sum()

            # actions.append(action)
        # actions = torch.stack(actions)
        # print(actions)
        return action.detach().numpy(), one_hot_action.detach().numpy(), log_probs

    def compute_loss(self, data, s_batch, a_batch, advantages, lp_batch, mode):
        assert mode == 'flow' or mode == 'node'
        actions = torch.Tensor(a_batch).detach().to(self.device)  # one-hot
        old_log_probs = torch.FloatTensor(lp_batch).detach().to(self.device)
        eps = 1e-9
        if mode == 'flow':
            logits = self.gcn.forward_flow(data)
        else:
            logits = self.gcn.forward_node(data)
        logits = logits.reshape((len(s_batch), -1))
        # probs = F.softmax(logits, dim=1)
        if (logits != logits).any():
            print(f'logits: {logits.tolist()}', flush=True)
        logits = torch.clamp(logits, min=-1e2, max=1e2)
        probs = torch.exp(F.log_softmax(logits, dim=1))
        if (probs != probs).any():
            print(f'probs: {probs.tolist()}', flush=True)
        m = Categorical(probs)
        entropy = m.entropy()
        probs = torch.unsqueeze(probs, dim=-1)
        log_probs = torch.log(torch.squeeze(
            torch.matmul(actions, probs)+eps)).sum(dim=1)
        # advantages = rewards - v_values.detach()
        # surrogate loss
        ratios = torch.exp(log_probs - old_log_probs.detach())
        ratios = torch.clamp(ratios, min=-1e2, max=1e2)
        # print(f'prob_diff: {log_probs - old_log_probs.detach()}')
        # print(f'ratios: {ratios}')
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.eps_clip,
                            1+self.eps_clip) * advantages
        s_loss = -torch.min(surr1, surr2).mean()
        # entropy loss
        e_loss = self.entropy_lr * entropy.mean()
        # print(f"s:{s_loss} e:{e_loss}")
        # total_loss = s_loss - e_loss
        return s_loss, e_loss

    def compute_MGDA_loss_scale(self, data, s_batch, advantages, a_batch1, a_batch2, lp_batch1, lp_batch2):
        grads = {}
        scale = {}
        loss_data = {}
        tasks = ('f', 'n')
        # grad of flow selection
        flow_loss, flow_e_loss = self.compute_loss(
            data, s_batch, a_batch1, advantages, lp_batch1, mode='flow')
        self.optim.zero_grad()
        flow_loss.backward()
        grads['f'] = []
        loss_data['f'] = flow_loss.mean().item()
        for param in self.gcn.parameters():
            if param.grad is not None:
                grads['f'].append(
                    Variable(param.grad.data.clone(), requires_grad=False))
        # grad of node selection
        node_loss, node_e_loss = self.compute_loss(
            data, s_batch, a_batch2, advantages, lp_batch2, mode='node')
        self.optim.zero_grad()
        node_loss.backward()
        grads['n'] = []
        loss_data['n'] = node_loss.mean().item()
        for param in self.gcn.parameters():
            if param.grad is not None:
                grads['n'].append(
                    Variable(param.grad.data.clone(), requires_grad=False))
        # normalizing
        # gn = gradient_normalizers(grads, loss_data, 'l2')
        # for t in tasks:
        #     for gr_i in range(len(grads[t])):
        #         grads[t][gr_i] = grads[t][gr_i] / gn[t]
        # MGDA
        sol, min_norm = MinNormSolver.find_min_norm_element(
            [grads[t] for t in tasks])
        for i, t in enumerate(tasks):
            scale[t] = float(sol[i])
        return scale

    def update(self, s_batch, a_batch1, a_batch2, r_batch, ad_batch, lp_batch1, lp_batch2, mode='both'):
        assert mode == 'both' or mode == 'flow' or mode == 'MGDA'
        # s_batch = torch.FloatTensor(s_batch).detach().to(self.device) # matrix
        # s_batch = ts_batch.to(self.device)
        loader = DataLoader(s_batch, batch_size=len(s_batch))
        rewards = torch.Tensor(r_batch).detach().to(self.device)
        advantages = torch.FloatTensor(ad_batch).detach().to(self.device)
        eps = 1e-9

        # reward normalization
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + eps)

        self.gcn.train()
        for data in loader:
            data = data.to(self.device)
            for i in range(self.k_epoch):
                if mode == 'MGDA':
                    scale = self.compute_MGDA_loss_scale(
                        data, s_batch, advantages, a_batch1, a_batch2, lp_batch1, lp_batch2)
                else:
                    scale = {'f': 1, 'fe': 1,
                             'n': 1, 'ne': 1}
                # print(f'epoch {i} loss : {total_loss1}')

                # compute loss
                self.optim.zero_grad()
                flow_loss, flow_e_loss = self.compute_loss(
                    data, s_batch, a_batch1, advantages, lp_batch1, mode='flow')
                if mode != 'flow':
                    node_loss, node_e_loss = self.compute_loss(
                        data, s_batch, a_batch2, advantages, lp_batch2, mode='node')
                else:
                    node_loss, node_e_loss = 0, 0
                total_loss = scale['f'] * flow_loss - scale['f'] * \
                    flow_e_loss + scale['n'] * \
                    node_loss - scale['n'] * node_e_loss
                total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.gcn.parameters(), 1.0)
                # torch.nn.utils.clip_grad_value_(self.gcn.parameters(), 1.0)
                self.optim.step()
        # self.scheduler.step()
        lr = self.scheduler.get_last_lr()

        # print(
        # f'flow loss: {flow_loss.mean()} node loss: {node_loss.mean() if node_loss != 0 else 0} weight: {scale} total: {total_loss} lr: {lr}', flush=True)
        print(
            f'flow loss: {flow_loss.mean()} flow entropy: {flow_e_loss.mean()} node loss: {node_loss.mean() if node_loss != 0 else 0} node entropy: {node_e_loss.mean() if node_e_loss != 0 else 0} weight: {scale} total: {total_loss} lr: {lr}', flush=True)

    def get_parameters(self):
        return self.gcn.state_dict()

    def set_parameters(self, state_dict):
        self.gcn.load_state_dict(state_dict)

    def save_parameters(self, name):
        torch.save(self.gcn.state_dict(), f'{name}')

    def load_parameters(self, name):
        self.gcn.load_state_dict(
            torch.load(f'{name}', map_location=torch.device('cpu')), strict=False
        )

    def load_parameters_from_trained_ns_model(self, name):
        if 'mlu' in config.lp_model:
            ns_dict = torch.load(f'{name}-mlu-PPOGCN.pkl',
                                 map_location=torch.device('cpu'))
        else:
            ns_dict = torch.load(f'{name}-mt-PPOGCN.pkl',
                                 map_location=torch.device('cpu'))

        for key in list(ns_dict.keys()):
            if 'fc' in key:
                ns_dict[key.replace('fc', 'fc2')] = ns_dict.pop(key)
        self.gcn.load_state_dict(ns_dict, strict=False)
