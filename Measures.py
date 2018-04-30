import numpy as np


def defaults_cdf(net):
    defaulting = net.get_defaulting_record()
    cum_defaulting = np.cumsum(np.array([np.sum(defaulting[:, t]) for t in range(0, defaulting.shape[1])]))
    return cum_defaulting


def cumulated_losses(net):
    return np.cumsum(net.get_lost_value())


def average_defaults_cdf(net_list):
    n_mc = len(net_list)
    T = len(net_list[0].get_lost_value())
    stack = np.empty((n_mc, T), dtype=np.float64)
    for s in range(0, n_mc):
        stack[s, :] = defaults_cdf(net_list[s])
    return stack.mean(axis=0)


def average_losses_cdf(net_list):
    n_mc = len(net_list)
    T = len(net_list[0].get_lost_value())
    stack = np.empty((n_mc, T), dtype=np.float64)
    for s in range(0, n_mc):
        stack[s, :] = cumulated_losses(net_list[s])
    return stack.mean(axis=0)


def average_losses_cdf_dict(dict_net_list):
    cdf_dict = {k: average_losses_cdf(v) for k, v in dict_net_list.items()}
    return cdf_dict


def average_defaults_cdf_dict(dict_net_list):
    cdf_dict = {k: average_defaults_cdf(v) for k, v in dict_net_list.items()}
    return cdf_dict