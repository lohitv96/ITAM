import numpy as np


def srm(w, S, t):
    Nw = len(w)
    Nt = len(t)
    dw = w[1] - w[0]
    phi = np.random.uniform(size=Nw) * 2 * np.pi
    a_t = np.zeros(shape=Nt)

    for i in range(Nt):
        a_t[i] = 2 * np.matrix(np.sqrt(dw * S)) * np.transpose(np.matrix(np.cos(w * t[i] + phi)))

    return a_t
