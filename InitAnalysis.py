import numpy as np

import BalanceSheetInit as BSI


def random_allocation(graph, p, vals, distrib):
    bers = graph.generate_dibernouilli(p)
    vals = graph.generate_loans_values(vals, distrib)
    graph.set_loans(bers * vals)
    # Initialization of the balance sheets
    L = graph.get_loans_matrix()
    return L


def init_generator (params_dict, x0, mus, L):
    init_bs = BSI.BalanceSheetInit(L,
                                   params_dict["r"],
                                   params_dict["q"],
                                   params_dict["alphas"],
                                   params_dict["betas"],
                                   params_dict["lambda_star"],
                                   x0,
                                   mus)
    E = params_dict["e"] * np.ones((params_dict["n"],))
    init_bs.set_manual_equities(E)
    return init_bs


def mc_on_initializations(params_dict, x0, mus, graph, p, vals, distrib, n_mc):
    init_list = []
    for i in range(0, n_mc):
        L = random_allocation(graph, p, vals, distrib)
        init_bs = init_generator(params_dict, x0, mus, L)
        init_list.append(init_bs)
    return init_list


def compare_initializations(params_dict,
                                  x0,
                                  mus,
                                  graph_dict,
                                  n_mc,
                                  p,
                                  vals,
                                  distrib):
    simus_dict = dict()
    for key in graph_dict.keys():
        simus_dict[key] = mc_on_initializations(params_dict, x0, mus, graph_dict[key], p, vals, distrib, n_mc)
    return simus_dict


