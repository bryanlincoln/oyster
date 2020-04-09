"""
Common evaluation utilities.
"""

from collections import OrderedDict
from numbers import Number
import os
import numpy as np

from . import pythonplusplus as ppp
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def dprint(*args):
    # hacky, but will do for now
    if int(os.environ['DEBUG']) == 1:
        print(args)


def get_generic_path_information(paths, stat_prefix=''):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]

    rewards = np.vstack([path["rewards"] for path in paths])
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))
    actions = [path["actions"] for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["actions"] for path in paths])
    else:
        actions = np.vstack([path["actions"] for path in paths])
    statistics.update(create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))
    statistics['Num Paths'] = len(paths)

    return statistics

def get_env_agent_path_information(paths, statistics, stat_prefix=''):
    # paths[ trajectories [ env_info { success } ] ]
    # for info_key in ['env_infos' , 'agent_infos']: # not really needed for now
    info_key = 'env_infos'
    if info_key in paths[0][0]: # if first trajectory of first path has info key
        all_env_infos = []
        for path in paths:
            for trajectory in path:
                all_env_infos.append(
                    ppp.list_of_dicts_to_dict_of_lists(
                        trajectory[info_key],
                        ['reachDist', 'goalDist', 'pickRew', 'epRew', 'success']
                    )
                )

        for k in all_env_infos[0].keys():
            final_ks = np.array([info[k][-1] for info in all_env_infos])
            first_ks = np.array([info[k][0] for info in all_env_infos])
            all_ks = np.concatenate([info[k] for info in all_env_infos])
            statistics.update(create_stats_ordered_dict(
                stat_prefix + '/' + k,
                final_ks,
                stat_prefix='{}/final/'.format(info_key),
            ))
            statistics.update(create_stats_ordered_dict(
                stat_prefix + '/' + k,
                first_ks,
                stat_prefix='{}/initial/'.format(info_key),
            ))
            statistics.update(create_stats_ordered_dict(
                stat_prefix + '/' + k,
                all_ks,
                stat_prefix='{}/'.format(info_key),
            ))

    return statistics


def get_average_returns(paths):
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    data = [0 if di is None or isinstance(di, str) else di for di in data]
    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats

def make_embedding_plotter(path):
    def plot_embeddings(embeddings, tasks_train, tasks_eval, epoch):
        tasks = np.concatenate((tasks_train, tasks_eval), axis=0)
        embeddings = np.reshape(embeddings, (-1, embeddings.shape[-1])) # transform to [train_size + eval_size, embedding_size]

        pca = PCA(n_components=2)
        pca.fit(embeddings)
        embeddings = pca.transform(embeddings)

        embeddings_train = embeddings[:len(tasks_train), :]
        embeddings_eval = embeddings[len(tasks_train):, :]

        plt.scatter(embeddings_train[:, 0], embeddings_train[:, 1], c=tasks_train, cmap='rainbow', marker="o", s=50, label='Train')
        plt.clim(tasks.min(), tasks.max())
        plt.scatter(embeddings_eval[:, 0], embeddings_eval[:, 1], c=tasks_eval, cmap='rainbow', marker="+", s=50, label='Eval')
        plt.clim(tasks.min(), tasks.max())
        plt.colorbar().set_label('Task', rotation=270)
        plt.ylim(ymax=10, ymin=-10)
        plt.xlim(xmax=10, xmin=-10)
        plt.grid(color='gray', linestyle='dashed')
        plt.grid(color='gray', linestyle='dashed')
        plt.legend(loc='upper right')
        plt.title('Epoch ' + str(epoch))
        plt.savefig(os.path.join(path, 'epoch_{}.eps'.format(epoch)), format='eps')

    return plot_embeddings