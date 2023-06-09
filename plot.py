import matplotlib
import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plot_utility import unique_dashes
from lpclient import Method
from copy import deepcopy

sns.set_theme(style='whitegrid')

TNODES = {'GEANT': 4, 'germany50': 10, 'rf1755': 43, 'rf6461': 69}
PALETTE = deepcopy(sns.color_palette())
NAME_MAP = {'rand': 'RAND', 'sp': 'SP', 'str': 'STR',
            'deg': 'DEG', 'sns': 'SNS', 'opt': 'OPT', 'fan': 'FAN'}


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('pgf')

# matplotlib.rcParams['ps.useafm'] = True
# matplotlib.rcParams['pdf.use14corefonts'] = True
# matplotlib.rcParams['text.usetex'] = True


class Plotter:
    def __init__(self, result_dir, fig_dir, num_inodes, num_tnodes, obj_kind):
        self.result_dir = result_dir
        self.fig_dir = fig_dir
        self.num_inodes = num_inodes
        self.num_tnodes = num_tnodes
        self.obj_kind = obj_kind  # MLU or MT

    def load_result(self, topo, method: Method):
        filename = f"{topo}-{method.to_str()}.pkl"
        file = open(f'{self.result_dir}/{filename}', 'rb')
        # { "id": ids, "obj": objs, "time": times }
        results = pickle.load(file)
        return results

    def y_label(self):
        if self.obj_kind == "MLU":
            return "Max Link Utilization"
        elif self.obj_kind == "MT":
            return "Demand Satisfaction"

    def all_methods(self):
        methods = []
        no_tnode_method = ['rand', 'sp', 'deg', 'str', 'sns']
        for name in no_tnode_method:
            m = Method(cd_method=name, num_inodes=self.num_inodes,
                       num_tnodes=None, lp_kind='LP', obj_kind=self.obj_kind)
            methods.append(m)
        fan = Method(cd_method='fan', num_inodes=self.num_inodes,
                     num_tnodes=self.num_tnodes, lp_kind='LP', obj_kind=self.obj_kind)
        opt = Method(cd_method='opt', num_inodes=None,
                     num_tnodes=None, lp_kind='LP', obj_kind=self.obj_kind)
        # methods.extend([fan, opt])
        methods.extend([opt])
        return methods

    def plot_umax_group_by_topo(self, topo):
        data = []
        methods = self.all_methods()
        for method in methods:
            # for method, result in results.items():
            result = self.load_result(topo, method)
            for (_, reward, time) in zip(result['id'], result['obj'], result['time']):
                data.append([NAME_MAP[method.cd_method], reward, time])

        df = pd.DataFrame(
            data, columns=['Method', 'Max Link Utilization', 'time'])
        ratio = df[df['Method'] == 'OURS'].mean() / df[df['Method']
                                                       == 'ALL'].mean()
        # print(f'ratio: {ratio}')
        fig = plt.figure()
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        sns.barplot(x="Method",
                    y="Max Link Utilization", data=df, capsize=.3)
        # sns.boxplot(x="method", y="max utilization", data=df)
        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/umax-{topo}.pdf", format="pdf")

    def plot_time_group_by_topo(self, topo):
        data = []
        methods = self.all_methods()
        for method in methods:
            # for method, result in results.items():
            result = self.load_result(topo, method)
            avg_time = np.mean(result['time'])
            print(f'Method:{method} Time:{avg_time} s')
            for (_, reward, time) in zip(result['id'], result['obj'], result['time']):
                data.append([NAME_MAP[method.cd_method], reward, time])

        df = pd.DataFrame(data, columns=['Method', 'max utilization', 'time'])
        fig = plt.figure()
        sns.boxplot(x="Method", y="time", data=df)
        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/time-{topo}.pdf", format="pdf")

    def plot_umax_and_time_group_by_topo(self, topo):
        data = []
        methods = self.all_methods()
        for method in methods:
            # for method, result in results.items():
            result = self.load_result(topo, method)
            for (_, reward, time) in zip(result['id'], result['obj'], result['time']):
                data.append([NAME_MAP[method.cd_method], reward, time])
        df = pd.DataFrame(
            data, columns=['Method', 'Max Link Utilization', 'Running Time'])

        for method in methods:
            print(f'topo: {topo} obj_kind: {self.obj_kind}')
            r = df[df['Method'] == NAME_MAP[method.cd_method]
                   ]['Max Link Utilization'].mean()
            t = df[df['Method'] == NAME_MAP[method.cd_method]
                   ]['Running Time'].mean()
            print(f'method: {method} res: {r} time: {t}')

        f, ax1 = plt.subplots()
        y_label = self.y_label()
        ax1.grid(True)
        # if "Demand" in y_label:
            # ax1.set_ylim(0.4, None)
        sns.barplot(x="Method",
                    y="Max Link Utilization", data=df, capsize=.3, ax=ax1, palette=PALETTE)
        df = pd.DataFrame(
            data, columns=['Method', 'Max Link Utilization', 'Running Time'])
        ax2 = ax1.twinx()
        estimator = getattr(np, 'mean')
        sns.pointplot(data=df, x="Method", y="Running Time", join=True, markers='o',
                      color=PALETTE[9],
                      ax=ax2, estimator=estimator)
        ax1.set_ylabel(y_label)
        ax2.set_ylabel("Running Time (s)")
        ax2.set_ylim(0, None)
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(-2, 4))
        print(df[df['Method'] == 'OURS']['Max Link Utilization'].mean())
        plt.tight_layout()
        plt.savefig(
            f"{self.fig_dir}/umax-time-{topo}-{self.obj_kind}.pdf", format="pdf")

    def plot_ratio_markers_group_by_topo(self, topo):
        # all methods
        methods = []
        no_tnode_method = ['rand', 'sp', 'deg', 'str', 'sns']
        for name in no_tnode_method:
            m = Method(cd_method=name, num_inodes=self.num_inodes,
                       num_tnodes=None, lp_kind='LP', obj_kind=self.obj_kind)
            methods.append(m)
        # fan = Method(cd_method='fan', num_inodes=self.num_inodes,
        #              num_tnodes=self.num_tnodes, lp_kind='LP', obj_kind=self.obj_kind)
        opt = Method(cd_method='opt', num_inodes=None,
                     num_tnodes=None, lp_kind='LP', obj_kind=self.obj_kind)
        # methods.extend([fan])

        result = self.load_result(topo, opt)
        idxes, rewards, times = result['id'], result['obj'], result['time']
        opt_result = {}
        for id, reward, time in zip(idxes, rewards, times):
            opt_result[id] = (reward, time)

        x = np.arange(0, 605, 1)
        fig, ax = plt.subplots(1, 1)
        plt.xlabel("Performance Ratio")
        plt.ylabel("CDF")
        dashes = unique_dashes(8)

        for mi, method in enumerate(methods):
            y = []
            result = self.load_result(topo, method)
            idxes, rewards, times = result['id'], result['obj'], result['time']
            for (idx, reward, time) in zip(idxes, rewards, times):
                full_reward, full_time = opt_result[idx]
                # if self.obj_kind == 'MLU':
                #     y.append(full_reward / reward)
                # else:
                y.append(reward / full_reward)
            y = np.array(y)
            # print(y)
            ymax = np.max(y)
            ys, bins, _ = plt.hist(y, bins=np.arange(0, ymax, 0.001),
                                   cumulative=True, density=True, histtype='step', linewidth=0)
            # plt.show()
            x = (bins[:-1] + bins[1:])/2
            plt.plot(x, ys, label=NAME_MAP[method.cd_method], linewidth=2.5, color=PALETTE[
                     mi], dashes=dashes[mi])
            plt.legend(title='Method')

        # ax.set_xbound(0.0, 1.0)
        ax.set_ybound(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(
            f"{self.fig_dir}/ratio-markers-{topo}-{self.obj_kind}.pdf", format="pdf")

    def plot_compare_i_it_and_opt(self, print_table=True):
        NAME_MAP = {'fan': 'SNS-T', 'sns': 'SNS-I',
                    'snsc': 'SNS-C', 'opt': 'OPT'}
        data = []
        for topo, num_tnodes in TNODES.items():
            snsm = Method(cd_method='sns', num_inodes=self.num_inodes,
                          num_tnodes=None, lp_kind='LP', obj_kind=self.obj_kind)
            fan = Method(cd_method='fan', num_inodes=self.num_inodes,
                         num_tnodes=num_tnodes, lp_kind='LP', obj_kind=self.obj_kind)
            snsc = Method(cd_method='snsc', num_inodes=self.num_inodes,
                          num_tnodes=num_tnodes, lp_kind='LP', obj_kind=self.obj_kind)
            for method in [snsm, fan, snsc]:
                # for method, result in results.items():
                result = self.load_result(topo, method)
                idxes, rewards, times = result['id'], result['obj'], result['time']
                for (_, reward, time) in zip(idxes, rewards, times):
                    data.append(
                        [NAME_MAP[method.cd_method], reward, time, topo])
        df = pd.DataFrame(
            data, columns=['Method', 'Max Link Utilization', 'Running Time', 'Topology'])

        if print_table:
            for method in ['sns', 'fan', 'snsc']:
                for column in ["Max Link Utilization", "Running Time", "Running Time"]:
                    for topo in TNODES.keys():
                        v = df[df['Topology'] == topo][df['Method']
                                                       == NAME_MAP[method]][column].mean()
                        print(f'& {v:.3f}', end=' ')
                print(r'\\')
            print()

        y_label = self.y_label()
        fig = plt.figure()
        fig.set_figwidth(10)
        fig.set_figheight(5)
        # f, ax1 = plt.subplots()
        # ax1.grid(True)
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        sns.set_palette(PALETTE)
        plot = sns.boxplot(hue="Method", x="Topology",
                           y="Max Link Utilization", data=df)
        plot.set_ylabel(y_label)
        # ax2 = ax1.twinx()
        # estimator = getattr(np, 'mean')
        # sns.pointplot(data=df, x="Method", y="Running Time", join=True, markers='o',
        #               color=PALETTE[9],
        #               ax=ax2, estimator=estimator, zorder=10)
        # # ax2.get_legend().set_visible(False)
        # ax2.set_ylabel("Running Time (s)")
        # ax2.set_ylim(0, None)
        # ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
        # print(df[df['Method'] == 'OURS']['Max Link Utilization'].mean())
        plt.tight_layout()
        plt.savefig(
            f"{self.fig_dir}/compare-sns-{self.obj_kind}.pdf", format="pdf")

    def plot_umax_and_time_group_by_topo_inodes(self, topo):
        data = []
        for k in range(2, 6+1):
            snsm = Method(cd_method='sns', num_inodes=k,
                          num_tnodes=None, lp_kind='LP', obj_kind=self.obj_kind)
            result = self.load_result(topo, snsm)
            idxes, rewards, times = result['id'], result['obj'], result['time']
            for (_, reward, time) in zip(idxes, rewards, times):
                data.append([NAME_MAP[snsm.cd_method], reward, time, k])
        # load optimal
        optm = Method(cd_method='opt', num_inodes=None,
                      num_tnodes=None, lp_kind='LP', obj_kind=self.obj_kind)
        result = self.load_result(topo, optm)
        idxes, rewards, times = result['id'], result['obj'], result['time']
        for (_, reward, time) in zip(idxes, rewards, times):
            data.append([NAME_MAP[optm.cd_method], reward, time, 'OPT'])
        df = pd.DataFrame(
            data, columns=['Method', 'Max Link Utilization', 'Running Time', 'k'])
        # fig = plt.figure()
        y_label = self.y_label()
        f, ax1 = plt.subplots()
        ax1.grid(True)
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        if "Max" in y_label:
            sns.set_palette("Blues")
            PP = PALETTE[1]
        else:
            sns.set_palette("YlOrBr")
            PP = PALETTE[0]
        sns.barplot(x="k",
                    y="Max Link Utilization", data=df, capsize=.3, ax=ax1)
        ax2 = ax1.twinx()
        estimator = getattr(np, 'mean')
        g = sns.pointplot(data=df, x="k", y="Running Time", join=True, markers='o',
                          color=PP,
                          ax=ax2, estimator=estimator)
        # g.set_yscale("log")
        # ax2.get_legend().set_visible(False)
        # sns.boxplot(x="method", y="max utilization", data=df)
        ax1.set_ylabel(y_label)
        ax1.set_xlabel(r"$K_I$")
        ax2.set_ylabel("Running Time (s)")
        ax2.set_ylim(0, None)
        # print(df[df['Method'] == 'OURS']['Max Link Utilization'].mean())
        plt.tight_layout()
        plt.savefig(
            f"{self.fig_dir}/umax-time-k-{topo}-{self.obj_kind}.pdf", format="pdf")

    def plot_umax_and_time_group_by_topo_tnodes(self, topo):
        data = []
        tnode_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        tnodes_settings = [2, 4, 6, 8, 11] if topo == "GEANT" else [
            8, 17, 26, 34, 43] if topo == "rf1755" else []
        for tnode_ratio, num_tnodes in zip(tnode_ratios, tnodes_settings):
            fan = Method(cd_method='fan', num_inodes=self.num_inodes,
                         num_tnodes=num_tnodes, lp_kind='LP', obj_kind=self.obj_kind)
            result = self.load_result(topo, fan)
            idxes, rewards, times = result['id'], result['obj'], result['time']
            for (_, reward, time) in zip(idxes, rewards, times):
                data.append([NAME_MAP[fan.cd_method],
                            reward, time, tnode_ratio])

        # load tk=1.0
        snsm = Method(cd_method='sns', num_inodes=self.num_inodes,
                      num_tnodes=None, lp_kind='LP', obj_kind=self.obj_kind)
        result = self.load_result(topo, snsm)
        idxes, rewards, times = result['id'], result['obj'], result['time']
        for (_, reward, time) in zip(idxes, rewards, times):
            data.append([NAME_MAP[snsm.cd_method], reward, time, "1.0"])

        # load optimal
        optm = Method(cd_method='opt', num_inodes=None,
                      num_tnodes=None, lp_kind='LP', obj_kind=self.obj_kind)
        result = self.load_result(topo, optm)
        idxes, rewards, times = result['id'], result['obj'], result['time']
        for (_, reward, time) in zip(idxes, rewards, times):
            data.append([NAME_MAP[optm.cd_method], reward, time, "OPT"])

        # plot
        y_label = self.y_label()
        df = pd.DataFrame(
            data, columns=['Method', 'Max Link Utilization', 'Running Time', 'tk'])
        # fig = plt.figure()
        f, ax1 = plt.subplots()
        ax1.grid(True)
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        if "Max" in y_label:
            sns.set_palette("Blues", n_colors=7)
            PP = PALETTE[1]
        else:
            sns.set_palette("YlOrBr", n_colors=7)
            PP = PALETTE[0]
        sns.barplot(x="tk",
                    y="Max Link Utilization", data=df, capsize=.3, ax=ax1)
        ax2 = ax1.twinx()
        estimator = getattr(np, 'mean')
        g = sns.pointplot(data=df, x="tk", y="Running Time", join=True, markers='o',
                          color=PP,
                          ax=ax2, estimator=estimator)
        # g.set_yscale("log")
        # ax2.get_legend().set_visible(False)
        # sns.boxplot(x="method", y="max utilization", data=df)
        ax1.set_ylabel(y_label)
        ax1.set_xlabel(r"$K_T/N$")
        ax2.set_ylabel("Running Time (s)")
        ax2.set_ylim(0, None)
        plt.tight_layout()
        plt.savefig(
            f"{self.fig_dir}/umax-time-tk-{topo}-{self.obj_kind}.pdf", format="pdf")
    

    def print_ratios_compare_to_opt(self):
        NAME_MAP = {'fan': 'SNS-T', 'sns': 'SNS-I',
                    'snsc': 'SNS-C', 'opt': 'OPT'}

        for topo, num_tnodes in TNODES.items():
            data = []
            # 1. load opt result
            opt = Method(cd_method='opt', num_inodes=None,
                         num_tnodes=None, lp_kind='LP', obj_kind=self.obj_kind)
            result = self.load_result(topo, opt)
            idxes, rewards, times = result['id'], result['obj'], result['time']
            opt_result = {}
            for id, reward, time in zip(idxes, rewards, times):
                opt_result[id] = { 'obj': reward, 'time': time }

            # 2. load our result
            snsm = Method(cd_method='sns', num_inodes=self.num_inodes,
                          num_tnodes=None, lp_kind='LP', obj_kind=self.obj_kind)
            fan = Method(cd_method='fan', num_inodes=self.num_inodes,
                         num_tnodes=num_tnodes, lp_kind='LP', obj_kind=self.obj_kind)
            snsc = Method(cd_method='snsc', num_inodes=self.num_inodes,
                          num_tnodes=num_tnodes, lp_kind='LP', obj_kind=self.obj_kind)
            for method in [snsm, fan, snsc]:
                # for method, result in results.items():
                result = self.load_result(topo, method)
                idxes, rewards, times = result['id'], result['obj'], result['time']
                
                for (id, reward, time) in zip(idxes, rewards, times):
                    opt_reward, opt_time = opt_result[id]['obj'], opt_result[id]['time']
                    data.append(
                        [NAME_MAP[method.cd_method], reward / opt_reward, time / opt_time])

            df = pd.DataFrame(
                data, columns=['Method', 'Obj Ratio', 'Time Ratio'])
            
            for column in ["Obj Ratio", "Time Ratio"]:
                # for method in ['sns', 'fan', 'snsc']:
                for method in ['sns']:
                    max = df[df['Method'] == NAME_MAP[method]][column].max()
                    min = df[df['Method'] == NAME_MAP[method]][column].min()
                    mean = df[df['Method'] == NAME_MAP[method]][column].mean()
                    print(f'Topo: {topo} Method: {NAME_MAP[method]} Col: {column} Max: {max}, Min: {min}, Avg: {mean}')


def plot_all_figures(base_num_inodes, obj_kind):
    num_inodes = base_num_inodes

    # 1. compare num_inodes and num_tnodes
    plotter = Plotter(result_dir='./result', fig_dir='./fig',
                      num_inodes=num_inodes, num_tnodes=None, obj_kind=obj_kind)
    plotter.plot_umax_and_time_group_by_topo_inodes("GEANT")
    plotter.plot_umax_and_time_group_by_topo_inodes("rf1755")
    plotter = Plotter(result_dir='./result', fig_dir='./fig',
                      num_inodes=num_inodes, num_tnodes=None, obj_kind=obj_kind)
    plotter.plot_umax_and_time_group_by_topo_tnodes("GEANT")
    plotter.plot_umax_and_time_group_by_topo_tnodes("rf1755")

    # 2. plot bar and CDF
    for topo, num_tnodes in TNODES.items():
        plotter = Plotter(result_dir='./result', fig_dir='./fig',
                          num_inodes=num_inodes, num_tnodes=num_tnodes, obj_kind=obj_kind)
        plotter.plot_umax_and_time_group_by_topo(topo)  # bar
        plotter.plot_ratio_markers_group_by_topo(topo)  # CDF

    # 3. compare I, T and C
    plotter = Plotter(result_dir='./result', fig_dir='./fig',
                      num_inodes=num_inodes, num_tnodes=None, obj_kind=obj_kind)
    plotter.plot_compare_i_it_and_opt()


def print_performance_compare(num_inodes, obj_kind):
    plotter = Plotter(result_dir='./result', fig_dir='./fig',
                      num_inodes=num_inodes, num_tnodes=None, obj_kind=obj_kind)
    plotter.print_ratios_compare_to_opt()

if __name__ == '__main__':
    sns.set(font_scale=1.5)
    sns.set_style("white")

    plot_all_figures(base_num_inodes=5, obj_kind="MLU")
    plot_all_figures(base_num_inodes=5, obj_kind="MT")

    print_performance_compare(num_inodes=5, obj_kind="MLU")
    print_performance_compare(num_inodes=5, obj_kind="MT")
