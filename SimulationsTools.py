import numpy as np
import BankNetwork
import importlib
import networkx as nx
import os
import pickle
import BalanceSheetInit as BSI
import GraphInit as GI
import RiskyAssets
import pathlib
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


def mc_on_graphs(params_dict, prices, x0, mus, graph, n_mc, p, vals, distrib):
    mc_list = []
    for s in range(0, n_mc):
        L = random_allocation(graph, p, vals, distrib)
        R, Q = balance_sheet_allocation(params_dict, L, x0, mus)
        # Creation of the bank network
        bank_network = BankNetwork.BankNetwork(L,
                                               R,
                                               Q,
                                               params_dict["alphas"],
                                               params_dict["r"],
                                               params_dict["xi"],
                                               params_dict["zeta"],
                                               params_dict["bar_E"])
        if params_dict["liquidator"]:
            bank_network.add_liquidator()
        bank_network.update_portfolios(prices[0, :])
        bank_network.compute_psi()
        bank_network.compute_pi()
        for t in range(0, params_dict["T"]):
            bank_network.stage1(prices[t, :])
            bank_network.stage2()
            bank_network.stage3()
            bank_network.snap_record()
        print(s)
        rec_tuple = (bank_network.get_defaulting_record(),
                     bank_network.get_indegree_record(),
                     bank_network.get_outdegree_record(),
                     bank_network.get_lost_value())
        mc_list.append(rec_tuple)
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


def mc_on_prices(params_dict, prices_list, x0, mus, graph, n_mc, p, vals, distrib, save_out=os.getcwd() + "/Simulations/", title=""):
    mc_list_prices = []
    count = 0
    pathlib.Path(save_out).mkdir(parents=True, exist_ok=True)
    for prices in prices_list:
        newlist = mc_on_graphs(params_dict, prices, x0, mus, graph, n_mc, p, vals, distrib)
        if n_mc == 1:
            pickle.dump(newlist[0], open(save_out + title + "_" + str(count) + ".pkl", "wb"))
        count += 1
        print("Prices iterations counter : " + str(count))
    return mc_list_prices


def mc_on_prices_ergraphs(params_dict,
        prices_list,
        x0,
        mus,
        er_ps,
        n_mc,
        p,
        vals,
        distrib):
    simus_dict = dict()
    for param in er_ps:
        graph = nx.erdos_renyi_graph(params_dict["n"], param)
        graph = GI.GraphInit(graph)
        simus_dict[param] = mc_on_prices(params_dict, prices_list, x0, mus, graph, n_mc, p, (2/param) * vals, distrib)
    return simus_dict


def compare_graphs(
        params_dict,
        prices,
        x0,
        mus,
        graph_dict,
        n_mc,
        p,
        vals,
        distrib):
    simus_dict = dict()
    for key in graph_dict.keys():
        simus_dict[key] = mc_on_graphs(
            params_dict,
            prices,
            x0,
            mus,
            graph_dict[key],
            n_mc,
            p,
            vals,
            distrib)
    return simus_dict


def compare_ER_graphs(
        params_dict,
        prices,
        x0,
        mus,
        er_ps,
        n_mc,
        p,
        vals,
        distrib):
    simus_dict = dict()
    for param in er_ps:
        graph = nx.erdos_renyi_graph(params_dict["n"], param)
        graph = GI.GraphInit(graph)
        simus_dict[param] = mc_on_graphs(
            params_dict, prices, x0, mus, graph, n_mc, p, vals, distrib)
    return simus_dict


