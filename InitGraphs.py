import numpy as np
import networkx
import scipy



testgraph = networkx.powerlaw_cluster_graph(10000, 100, 0.1)
adjacency = networkx.to_numpy_matrix(testgraph)
# networkx.draw(testgraph)

cycle = networkx.cycle_graph(10)
networkx.draw(cycle)

star = networkx.star_graph(10)
networkx.draw(star)

complete = networkx.complete_graph(10)
networkx.draw(complete)
a = networkx.to_numpy_matrix(complete)


def vec_gene(v):
    mat = np.zeros((v.shape[0] + 1, v.shape[0]))
    mat[0, :] = v
    mat[1:, :] = np.diag(v)
    return mat


def expanded_adjacency(A):
    n = A.shape[0]
    n_exp = (n * (n - 1)) / 2
    A_exp = np.zeros((n, int(n_exp)))
    col = 0
    for i in range(1, n):
        v = A[i - 1, i:]
        print(v)
        mat = vec_gene(v)
        A_exp[i - 1:, col : col + n - i] = mat
        col += n - i
    return A_exp


complete = networkx.complete_graph(5)
networkx.draw(complete)
A = networkx.to_numpy_matrix(complete)
A = np.asarray(A)

star = networkx.star_graph(10)
networkx.draw(star)


A_exp = expanded_adjacency(A)

Apinv = np.linalg.pinv(A_exp)

xtest = np.zeros((5, 1))
xtest[0, 0] = 100
xtest[1, 0] = - 100
xtest[2, 0] = 200
xtest[3, 0] = -100
xtest[4, 0] = 100

sol = np.dot(Apinv, xtest)
print(sol)