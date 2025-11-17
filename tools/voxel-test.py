import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x, y, z = np.random.rand(3, 100) * 4
hist, edges = np.histogramdd((x, y, z), bins=10)
xedges, yedges, zedges = edges

# Construct arrays for the anchor positions of the bars.
xpos, ypos, zpos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, zedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = zpos.ravel()

# Construct arrays with the dimensions for the bars.
dx = dy = dz_dim = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.show()