# FEniCS_Omega_h

FEniCS  simulations in fluid/solid mechanics with omega_h mesh adaptivity 

The forked Omega_h repository that contains all the modifications necessary to make the latest version v9.24.2, demos and examples work with the latest FEniCS version v2018.1.0 is available on github https://github.com/tamaradanceva/omega_h/tree/FEniCS, in the FEniCS branch.

## Poisson equation 

![Image of initial mesh - Poisson equation demo](poisson/init_mesh.png)

![Image of adapted mesh - Poisson equation demo](poisson/adapted_mesh.png)


## DFS simulation of turbulent flow past a NACA wing

![Image of initial mesh - DFS simulation of a NACA0012 wing](dfs_wing/init_dfs_mesh.png)

![Image of the adapted mesh and the velocity as a metric - DFS simulation of a NACA0012 wing](dfs_wing/metric_dfs_mesh.png)

![Image of the adapted mesh - DFS simulation of a NACA0012 wing](dfs_wing/wireframe_dfs_mesh.png)
