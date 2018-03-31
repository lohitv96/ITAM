from SRM import *
from collections import deque
from timeit import timeit
import matplotlib.pyplot as plt
import json

plt.style.use('seaborn')

# Input Data
# Time
dim = 2
T = 10
dt = 0.1
m = int(T / dt) + 1
t = np.linspace(0, T, m)

# Frequency
n = 100 + 1
W = np.array([1.5, 2.5])
dw = W / (n - 1)
x_list = [np.linspace(0, W[i], n) for i in range(dim)]
xy_list = np.array(np.meshgrid(*x_list, indexing='ij'))
S = 125 / 4 * np.linalg.norm(xy_list, axis=0) ** 2 * np.exp(-5 * np.linalg.norm(xy_list, axis=0))

json.dump(xy_list.tolist(), open('W_data.txt', 'w'))
json.dump(S.tolist(), open('S_data.txt', 'w'))
del xy_list
del S

f = open('W_data.txt')
xy_list = deque(f)
# xy_input = json.load(open('W_data.txt'))
# S = json.loads(open('S_data.txt'))

# SRM_object = SRM(1, S, dw, m, n, case='uni')
# samples = SRM_object.samples
