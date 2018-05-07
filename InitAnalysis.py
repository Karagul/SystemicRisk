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


def get_consolidated_equities(init_list):
    n = init_list[0].get_equities().shape[0]
    conso = np.zeros((n, len(init_list)))
    for i in range(0, len(init_list)):
        conso[:, i] = init_list[i].get_equities()
    return conso


def get_consolidated_loans(init_list):
    n = init_list[0].get_equities().shape[0]
    conso = np.zeros((n, len(init_list)))
    for i in range(0, len(init_list)):
        conso[:, i] = init_list[i].get_loans()
    return conso


def get_consolidated_debts(init_list):
    n = init_list[0].get_equities().shape[0]
    conso = np.zeros((n, len(init_list)))
    for i in range(0, len(init_list)):
        conso[:, i] = init_list[i].get_debts()
    return conso


def get_consolidated_assets(init_list):
    n = init_list[0].get_equities().shape[0]
    conso = np.zeros((n, len(init_list)))
    for i in range(0, len(init_list)):
        conso[:, i] = init_list[i].get_assets()
    return conso


def get_consolidated_reserves(init_list):
    n = init_list[0].get_equities().shape[0]
    conso = np.zeros((n, len(init_list)))
    for i in range(0, len(init_list)):
        conso[:, i] = init_list[i].get_reserves()
    return conso


def get_consolidated_portfolios(init_list):
    n = init_list[0].get_equities().shape[0]
    conso = np.zeros((n, len(init_list)))
    for i in range(0, len(init_list)):
        conso[:, i] = init_list[i].get_portfolios()
    return conso


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


def get_consodict_equities(comp_dict):
    simus_dict = {key: get_consolidated_equities(comp_dict[key]).flatten() for key in comp_dict.keys()}
    return simus_dict


def get_consodict_assets(comp_dict):
    simus_dict = {key: get_consolidated_assets(comp_dict[key]).flatten() for key in comp_dict.keys()}
    return simus_dict


def get_consodict_reserves(comp_dict):
    simus_dict = {key: get_consolidated_reserves(comp_dict[key]).flatten() for key in comp_dict.keys()}
    return simus_dict


def get_consodict_portfolios(comp_dict):
    simus_dict = {key: get_consolidated_portfolios(comp_dict[key]).flatten() for key in comp_dict.keys()}
    return simus_dict


def get_consodict_loans(comp_dict):
    simus_dict = {key: get_consolidated_loans(comp_dict[key]).flatten() for key in comp_dict.keys()}
    return simus_dict


def get_consodict_debts(comp_dict):
    simus_dict = {key: get_consolidated_debts(comp_dict[key]).flatten() for key in comp_dict.keys()}
    return simus_dict