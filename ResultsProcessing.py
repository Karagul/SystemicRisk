import pickle
import os
import numpy as np


def defaulting_from_file(file):
    content = pickle.load(open(file, "rb"))
    n = content[0]["Defaulting"].shape[0]
    T = len(content)
    d = np.zeros((n, T))
    count = 0
    for i in range(0, T):
        d[:, count] = content[i]["Defaulting"]
    return d


def cdf_defaulting_from_file(file):
    content = pickle.load(open(file, "rb"))
    cdf_defaulting = np.cumsum(np.sum(content[0], axis=0))
    return cdf_defaulting


def cum_lost_from_file(file):
    content = pickle.load(open(file, "rb"))
    lost_value = np.cumsum(np.array(content[-1]))
    return lost_value


def avg_indegree_from_file(file):
    content = pickle.load(open(file, "rb"))
    avg_indegree = np.mean(content[1], axis=0)
    return avg_indegree


def in_degrees_from_files(file):
    content = pickle.load(open(file, "rb"))
    return content[1]


def mc_indegree_from_folder(folder):
    first = True
    count = 0
    for file in os.listdir(folder):
        indegree = in_degrees_from_files(folder + file)
        if first:
            avg = indegree.copy()
            first = False
        else :
            avg += cdf_defaulting
        count += 1
    return (1 / count) * avg


def average_cdf_defaulting_from_folder(folder):
    first = True
    count = 0
    for file in os.listdir(folder):
        cdf_defaulting = cdf_defaulting_from_file(folder + file)
        if first :
            avg = cdf_defaulting.copy()
            first = False
        else :
            avg += cdf_defaulting
        count += 1
    return (1 / count) * avg


def average_lost_from_folder(folder):
    first = True
    count = 0
    for file in os.listdir(folder):
        print(file)
        lost = cum_lost_from_file(folder + file)
        if first:
            avg = lost.copy()
            first = False
        else:
            avg += lost
        count += 1
    return (1 / count) * avg


def average_indegree_from_folder(folder):
    first = True
    count = 0
    for file in os.listdir(folder):
        avg_indegree = avg_indegree_from_file(folder + file)
        if first :
            avg = avg_indegree.copy()
            first = False
        else:
            avg += avg_indegree
        count += 1
    return (1 / count) * avg



folders_dict = dict()
cdfs_defaults_dict = dict()
indegree_dict = dict()
# er_ps = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
er_ps = [0.005, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.09, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]


# cdfs_defaults_dict["0.5"] = average_cdf_defaulting_from_folder(folders_dict["0.5"])

for p in er_ps:
    folders_dict[str(p)] = "E:/Simulations/ER" + str(p) + "_Leverage10/"
    cdfs_defaults_dict[str(p)] = average_cdf_defaulting_from_folder(folders_dict[str(p)])
    print(p)

for p in er_ps:
    indegree_dict[str(p)] = average_indegree_from_folder(folders_dict[str(p)])
    print(p)


for p in er_ps:
    pickle.dump( cdfs_defaults_dict[str(p)], open("E:/Simulations/" + str(p) + ".pkl", "wb"))
    print(p)


pickle.dump( cdfs_defaults_dict["0.5"], open("E:/Simulations/" + "0.5" + ".pkl", "wb"))
cdf_defaulting01 = average_cdf_defaulting_from_folder(folder01)
cdf_defaulting001 = average_cdf_defaulting_from_folder(folder001)
cdf_defaulting05 = average_cdf_defaulting_from_folder(folder05)

p_ers_bis = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
for p in p_ers_bis[::-1]:
    plt.plot(cdfs_defaults_dict[str(p)], label=str(p))
plt.legend()

plt.title("Averaged over 1000 runs - ER graphs parameters comparison")
plt.xlabel("Period")
plt.ylabel("Cumulative number of defaults")


plt.plot(cdf_defaulting01, label="0.1")
plt.plot(cdf_defaulting05, label="0.5")
plt.plot(cdf_defaulting001, label="0.01")
plt.legend()
plt.show()

cdf_lost = average_lost_from_folder(folder)

cdf_defaulting001 = cdf_defaulting


plt.plot(cdf_defaulting01, label="0.1")

file = "E:/Simulations/ER0.5_Leverage10_2.pkl"

content = pickle.load(open(file, "rb"))

defaulting = content[0]
losses = content[1]

cdf_defaulting = np.cumsum(np.sum(defaulting, axis=0))