#!/usr/bin/env python
"""
FEniCS code for -div(-A(x/eps)u) = f in 1D
Exact solving for fixed eps
"""

from __future__ import print_function
import numpy as np
import fenics as fe

# Create mesh and define function space
mesh = fe.UnitIntervalMesh(800)
V = fe.FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = fe.Constant(0.0)


def boundary(x, on_boundary):
    return on_boundary


bc = fe.DirichletBC(V, u_D, boundary)

# Define variational problem
eps = fe.Constant(1E-1)
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
f = fe.Constant(1)
A = fe.Expression('1./(2+cos(2*pi*x[0]/eps))', degree=2, eps=eps)
a = A*fe.dot(fe.grad(u), fe.grad(v)) * fe.dx
L = f * v * fe.dx

# Compute solution
u = fe.Function(V)
fe.solve(a == L, u, bc)

# Plot solution
fe.plot(u)
fe.plot(A,mesh=mesh)

# # Save solution to file in VTK format
# vtkfile = fe.File('poisson/solution.pvd')
# vtkfile << u

# Hold plot
fe.interactive()
