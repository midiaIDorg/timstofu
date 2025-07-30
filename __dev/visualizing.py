import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_voxels(grid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    filled = np.argwhere(grid >= 0)
    for x, y, z in filled:
        ax.bar3d(x, y, z, 1, 1, 1, color=plt.cm.tab10(grid[x, y, z] % 10))
    plt.show()


grid = np.random.randint(0, 5, size=(5, 5, 5))
grid[0] = 0
plot_voxels(grid)
