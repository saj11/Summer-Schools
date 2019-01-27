from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
ax = fig.gca(projection='3d')
ax2 = fig2.gca(projection='3d')
ax3 = fig3.gca(projection='3d')

# Make data.
X = np.arange(0, 2, 0.01)
Y = np.arange(0, 2, 0.01)
X, Y = np.meshgrid(X, Y)
#dot product for 2d vector
Z = np.multiply(X,X) + np.multiply(Y,Y);

# Plot the surface.
surf = ax.plot_surface(X, Y, Z)
# Customize the z axis.
ax.set_zlim(0, 5)

#analytical gradient
Xag = 2 * X;
Yag = 2 * Y;


#numerical gradient
h = 0.01 #dx step
gy, gx = np.gradient(Z, h)

# Plot the surface.
surf2 = ax2.plot_surface(X, Y, Xag)
# Customize the z axis.
ax2.set_zlim(0, 5)

# Plot the surface.
surf3 = ax3.plot_surface(X, Y, gx)
# Customize the z axis.
ax3.set_zlim(0, 5)

