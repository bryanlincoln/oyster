"""
Common evaluation utilities.
"""

from collections import OrderedDict
from numbers import Number
import os
import numpy as np


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
    # if success in path -> trajectory -> env_info -> success
    if 'success' in paths[0][0]['env_infos'][0]:
        successes = []
        for path in paths:
            for trajectory in path:
                successes.append(trajectory['env_infos'][-1]['success'])
        statistics.update(create_stats_ordered_dict(
            stat_prefix + '/success',
            np.array(successes),
            stat_prefix='{}/'.format('env_infos'),
        ))

    """ TODO add more statistics
    for info_key in ['env_infos', 'agent_infos']:
        # if key exists in at least one path, it exists in all of them
        if info_key in paths[0]:
            # convert a list (each path) of dicts (path's info)
            # to a dict (info) of lists (info about all paths)
            all_env_infos = {}
            for path in paths:
                for info in path[info_key]:
                    if type(all_env_infos[info]) is list:
                        all_env_infos[info].append(path[info_key][info])
                    else:
                        all_env_infos[info] = [path[info_key][info]]
            for k in all_env_infos.keys():
                final_ks = np.array([info[k][-1] for info in all_env_infos])
                first_ks = np.array([info[k][0] for info in all_env_infos])
                all_ks = np.concatenate([info[k] for info in all_env_infos])
                statistics.update(create_stats_ordered_dict(
                    stat_prefix + k,
                    final_ks,
                    stat_prefix='{}/final/'.format(info_key),
                ))
                statistics.update(create_stats_ordered_dict(
                    stat_prefix + k,
                    first_ks,
                    stat_prefix='{}/initial/'.format(info_key),
                ))
                statistics.update(create_stats_ordered_dict(
                    stat_prefix + k,
                    all_ks,
                    stat_prefix='{}/'.format(info_key),
                ))
            """

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
