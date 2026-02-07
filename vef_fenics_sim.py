The Continuum Field Solver (vef_fenics_sim.py)
This script utilizes the FEniCS framework to model the Negative Field (NF) density as a continuous fluid substrate.

Python
￼
from dolfin import *
import numpy as np

# 1. Mesh and Function Space
mesh = UnitCubeMesh(24, 24, 24)
V = FunctionSpace(mesh, 'P', 1)

# 2. Parameters
kappa = Constant(150.0 * 0.656) # NF Tension scaled by D_eff
psi_base = Constant(1.0)        # Background NF equilibrium
sigma = 0.05                    # Gaussian width of nodes

# 3. PP Source Term (The 8-node lattice)
# Representing nodes as Gaussian voids in the NF
pp_source_expr = ""
nodes = [[0.2, 0.2, 0.2], [0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.8, 0.8, 0.2],
         [0.2, 0.2, 0.8], [0.8, 0.2, 0.8], [0.2, 0.8, 0.8], [0.8, 0.8, 0.8]]

for i, p in enumerate(nodes):
    term = f"exp(-(pow(x[0]-{p[0]},2)+pow(x[1]-{p[1]},2)+pow(x[2]-{p[2]},2))/({sigma}**2))"
    pp_source_expr += term + (" + " if i < len(nodes)-1 else "")

f_pp = Expression(pp_source_expr, degree=2)

# 4. Variational Problem (Static Equilibrium)
# -∇²ψ + κ(ψ - ψ_base) = f_pp
psi = TrialFunction(V)
v = TestFunction(V)
a = (dot(grad(psi), grad(v)) + kappa * psi * v) * dx
L = (f_pp + kappa * psi_base) * v * dx

# 5. Solve
psi_sol = Function(V)
solve(a == L, psi_sol)

# 6. Output Analysis
print(f"NF Lock Depth (Min Density): {psi_sol.vector().min():.4f}")
File(f"vef_field_3d.pvd") << psi_sol # For visualization in ParaView