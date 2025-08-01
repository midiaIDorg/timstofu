from __future__ import annotations

import numpy as np

import pyvista as pv

from pyvista import examples


# Download a volumetric dataset
vol = examples.download_knee_full()
vol

# A nice camera position
cpos = [(-381.74, -46.02, 216.54), (74.8305, 89.2905, 100.0), (0.23, 0.072, 0.97)]

vol.plot(volume=True, cmap="bone", cpos=cpos)


import numpy as np
import pyvista as pv

# Example 3D data


# Convert numpy array to pyvista UniformGrid
grid = pv.UniformGrid()

# Set grid dimensions: note that dimensions are one more than the number of cells in each direction
grid.dimensions = np.array(movie_frames.shape) + 1
grid.origin = (0, 0, 0)  # Origin of the grid
grid.spacing = (1, 1, 1)  # Voxel spacing

# Add the data to the cell data
grid.cell_data["values"] = data.flatten(order="F")  # Fortran order

# Plot the volume
grid.plot(volume=True)


import numpy as np
import pyvista

rng = np.random.default_rng(seed=0)
point_cloud = rng.random((100, 3))
pdata = pyvista.PolyData(point_cloud)
pdata["orig_sphere"] = np.arange(100)

# create many spheres from the point cloud
sphere = pyvista.Sphere(radius=0.02, phi_resolution=10, theta_resolution=10)
pc = pdata.glyph(scale=False, geom=sphere, orient=False)
pc.plot(cmap="Reds")

from __future__ import annotations

import numpy as np


from pyvista import examples


# Download a volumetric dataset
vol = examples.download_knee_full()
vol

# Create the spatial reference


# Set the grid dimensions: shape because we want to inject our values on the
#   POINT data
# values = np.linspace(0, 10, 1000).reshape((20, 5, 10))
values = movie_frames
