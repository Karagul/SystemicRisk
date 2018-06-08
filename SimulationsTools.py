""""*******************************************************
 * Copyright (C) 2017-2018 Dimitri Bouche dimi.bouche@gmail.com
 *
 * This file is part of an ongoing research project on systemic risk @CMLA
 *
 * This file can not be copied and/or distributed without the express
 * permission of Dimitri Bouche.
 *******************************************************"""


# Third party imports
import numpy as np
import importlib
import networkx as nx

# Local imports
import BankNetwork
import BalanceSheetInit as BSI
import GraphInit as GI
import RiskyAssets
importlib.reload(BankNetwork)
importlib.reload(GI)
importlib.reload(BSI)


def daily_compound(r, nperiods):
    return np.power(1 + r, 1 / nperiods) - 1


def random_allocation(graph, p, vals, distrib):
    bers = graph.generate_dibernouilli(p)
    vals = graph.generate_loans_values(vals, distrib)
    graph.set_loans(bers * vals)
    # Initialization of the balance sheets
    L = graph.get_loans_matrix()
    return L


def er_random_allocation(p_er, p, vals, distrib, n):
    graph = nx.erdos_renyi_graph(n, p_er)
    # graph = nx.complete_graph(n)
    graph = GI.GraphInit(graph)
    bers = graph.generate_dibernouilli(p)
    vals = graph.generate_loans_values(vals, distrib)
    graph.set_loans(bers * vals)
    # Initialization of the balance sheets
    L = graph.get_loans_matrix()
    return L


def balance_sheet_allocation(params_dict, L, x0, mus):
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
    R = init_bs.get_reserves()
    Q = init_bs.get_quantitities()
    return R, Q


def iterate_periods(params_dict, prices, x0, mus, graph, p, vals, distrib, last_L=False):
    L = random_allocation(graph, p, vals, distrib)
    R, Q = balance_sheet_allocation(params_dict, L, x0, mus)
    # Creation of the bank network
    bank_network = BankNetwork.BankNetwork(L, R, Q,
                                           params_dict["alphas"],
                                           params_dict["r"],
                                           params_dict["xi"],
                                           params_dict["zeta"],
                                           bar_E=params_dict["bar_E"],
                                           lambda_star=params_dict["lambda_star"],
                                           theta=params_dict["theta"],
                                           enforce_leverage=params_dict["enforce_leverage"])
    if params_dict["liquidator"]:
        bank_network.add_liquidator()
    bank_network.update_portfolios(prices[0, :])
    # print(bank_network.defaulting)
    # print(bank_network.get_equities())
    bank_network.update_defaulted()
    bank_network.update_defaulting()
    bank_network.compute_psi()
    bank_network.compute_pi()
    bank_network.set_initial_value()
    bank_network.record_defaults()
    bank_network.record_degrees()
    bank_network.record_assets()
    # print(bank_network.defaulting)
    # print(bank_network.get_equities())
    for t in range(1, params_dict["T"]):
        bank_network.stage1(prices[t, :])
        bank_network.stage2()
        bank_network.stage3()
        bank_network.record_defaults()
        bank_network.record_degrees()
    if last_L:
        rec_tuple = (bank_network.cumdefaults_leverage.copy(),
                     bank_network.cumdefaults_classic.copy(),
                     bank_network.get_normalized_cumlosses(),
                     bank_network.in_degrees.copy(),
                     bank_network.out_degrees.copy(),
                     bank_network.firesale_coefs.copy(),
                     bank_network.L.copy())
    else:
        rec_tuple = (bank_network.cumdefaults_leverage.copy(),
                     bank_network.cumdefaults_classic.copy(),
                     bank_network.get_normalized_cumlosses(),
                     bank_network.in_degrees.copy(),
                     bank_network.out_degrees.copy(),
                     bank_network.get_assets_record())
    return rec_tuple


def iterate_periods_er(params_dict, prices, x0, mus, p_er, p, vals, distrib, last_L=False):
    n = params_dict["n"]
    L = er_random_allocation(p_er, p, vals, distrib, n)
    R, Q = balance_sheet_allocation(params_dict, L, x0, mus)
    # Creation of bank network
    bank_network = BankNetwork.BankNetwork(L, R, Q,
                                           params_dict["alphas"],
                                           params_dict["r"],
                                           params_dict["xi"],
                                           params_dict["zeta"],
                                           bar_E=params_dict["bar_E"],
                                           lambda_star=params_dict["lambda_star"],
                                           theta=params_dict["theta"],
                                           enforce_leverage=params_dict["enforce_leverage"])
    if params_dict["liquidator"]:
        bank_network.add_liquidator()
    bank_network.update_portfolios(prices[0, :])
    bank_network.compute_psi()
    bank_network.compute_pi()
    bank_network.set_initial_value()
    bank_network.record_defaults()
    bank_network.record_degrees()
    bank_network.record_assets()
    for t in range(0, params_dict["T"] - 1):
        bank_network.stage1(prices[t, :])
        bank_network.stage2()
        bank_network.stage3()
        bank_network.record_defaults()
        bank_network.record_degrees()
    if last_L:
        rec_tuple = (bank_network.cumdefaults_leverage.copy(),
                     bank_network.cumdefaults_classic.copy(),
                     bank_network.get_normalized_cumlosses(),
                     bank_network.in_degrees.copy(),
                     bank_network.out_degrees.copy(),
                     bank_network.L.copy())
    else:
        rec_tuple = (bank_network.cumdefaults_leverage.copy(),
                     bank_network.cumdefaults_classic.copy(),
                     bank_network.get_normalized_cumlosses(),
                     bank_network.in_degrees.copy(),
                     bank_network.out_degrees.copy())
                     # bank_network.get_assets_record())
    return rec_tuple


def mc_on_graphs(params_dict, prices, x0, mus, graph, n_mc, p, vals, distrib):
    mc_list = []
    for s in range(0, n_mc):
        rec_tuple = iterate_periods(params_dict, prices, x0, mus, graph, p, vals, distrib)
        mc_list.append(rec_tuple)
        print(s)
    return mc_list


def mc_on_er_graphs(params_dict, prices, x0, mus, p_er, n_mc, p, vals, distrib):
    mc_list = []
    n = params_dict["n"]
    for s in range(0, n_mc):
        graph = nx.erdos_renyi_graph(n, p_er)
        graph = GI.GraphInit(graph)
        rec_tuple = iterate_periods(params_dict, prices, x0, mus, graph, p, 2 * vals / p_er, distrib)
        mc_list.append(rec_tuple)
        print(s)
    return mc_list


def generate_prices(x0, m, mu, sigma, T, nmc):
    prices_list = []
    for i in range(0, nmc):
        init_val = x0 * np.ones((m,))
        mus = mu * np.ones((m,))
        sigmas = sigma * np.ones((m,))
        assets = RiskyAssets.AdditiveGaussian(mus, sigmas, init_val, T)
        prices_list.append(assets.generate())
    return prices_list


def mc_on_prices(params_dict, prices_list, x0, mus, graph, p, vals, distrib):
    mc_list_prices = []
    counter = 0
    for prices in prices_list:
        mc_list_prices.append(iterate_periods(params_dict, prices, x0, mus, graph, p, vals, distrib))
        counter += 1
        print(counter - 1)
    return mc_list_prices


def mc_full_er(params_dict, prices_list, x0, mus, p_er, p, vals, distrib):
    mc_list_prices = []
    counter = 0
    n = params_dict["n"]
    for prices in prices_list:
        graph = nx.erdos_renyi_graph(n, p_er)
        graph = GI.GraphInit(graph)
        mc_list_prices.append(iterate_periods(params_dict, prices, x0, mus, graph, p, 2 * vals / p_er, distrib))
        counter += 1
        # print(counter - 1)
    return mc_list_prices

# def mc_on_prices(params_dict, prices_list, x0, mus, graph, n_mc, p, vals, distrib, save_out=os.getcwd() + "/Simulations/", title="", count_init=0):
#     mc_list_prices = []
#     count = count_init
#     pathlib.Path(save_out).mkdir(parents=True, exist_ok=True)
#     for prices in prices_list:
#         newlist = mc_on_graphs(params_dict, prices, x0, mus, graph, n_mc, p, vals, distrib)
#         if n_mc == 1:
#             pickle.dump(newlist[0], open(save_out + title + "_" + str(count) + ".pkl", "wb"))
#         count += 1
#         print("Prices iterations counter : " + str(count))
#     return mc_list_prices


# def mc_on_prices_ergraphs(params_dict,
#         prices_list,
#         x0,
#         mus,
#         er_ps,
#         n_mc,
#         p,
#         vals,
#         distrib):
#     simus_dict = dict()
#     for param in er_ps:
#         graph = nx.erdos_renyi_graph(params_dict["n"], param)
#         graph = GI.GraphInit(graph)
#         simus_dict[param] = mc_on_prices(params_dict, prices_list, x0, mus, graph, n_mc, p, (2/param) * vals, distrib)
#     return simus_dict
#
#
# def compare_graphs(
#         params_dict,
#         prices,
#         x0,
#         mus,
#         graph_dict,
#         n_mc,
#         p,
#         vals,
#         distrib):
#     simus_dict = dict()
#     for key in graph_dict.keys():
#         simus_dict[key] = mc_on_graphs(
#             params_dict,
#             prices,
#             x0,
#             mus,
#             graph_dict[key],
#             n_mc,
#             p,
#             vals,
#             distrib)
#     return simus_dict
#
#
# def compare_ER_graphs(
#         params_dict,
#         prices,
#         x0,
#         mus,
#         er_ps,
#         n_mc,
#         p,
#         vals,
#         distrib):
#     simus_dict = dict()
#     for param in er_ps:
#         graph = nx.erdos_renyi_graph(params_dict["n"], param)
#         graph = GI.GraphInit(graph)
#         simus_dict[param] = mc_on_graphs(
#             params_dict, prices, x0, mus, graph, n_mc, p, vals, distrib)
#     return simus_dict
#

