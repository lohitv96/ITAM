import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.special import sph_harm

########################################################################################################################
# Code for simulation of Gaussian Random Fields on a sphere from an angular power spectrum
# A very promising venue to be able to extend the higher-order spectral representation methods we have developed.
# Could possible help to expand them to simulating on manifolds.
########################################################################################################################

phi = np.linspace(0, np.pi, 100)
theta = np.linspace(0, 2*np.pi, 100)
phi, theta = np.meshgrid(phi, theta)

# The Cartesian coordinates of the unit sphere
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# l is the number of spherical frequencies to be considered
l_end = 25

l_list = np.arange(l_end)
c_l = np.exp(-l_list)

for l in l_list:
    for m in range(l):
        print(l, m)

# Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
fcolors = sph_harm(m, l, theta, phi).real
fmax, fmin = fcolors.max(), fcolors.min()
fcolors = (fcolors - fmin)/(fmax - fmin)
