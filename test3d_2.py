from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')

color = 'r'
x = np.random.sample(20)
y = np.random.sample(20)
x = [0]
y = [0]
print(x)
ax.scatter(x, y, 0, c=color)

ax.legend()
ax.set_xlim3d(0, 1)
ax.set_ylim3d(0, 1)
ax.set_zlim3d(0, 1)

plt.show()
