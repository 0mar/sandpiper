#!/usr/bin/env python
"""
FEniCS code for -div(-A(x/eps)u) = f in 1D
Unhomogenized solution approximation for fixed eps > 0
"""

from __future__ import print_function
import numpy as np
import fenics as fe
import matplotlib.pyplot as plt


def solve_poisson_eps(h, eps, plot=False):
    eps = fe.Constant(eps)
    n = int(1 / h)
    # Create mesh and define function space
    mesh = fe.UnitIntervalMesh(n)
    V = fe.FunctionSpace(mesh, 'P', 1)

    # Define boundary condition
    u_D = fe.Constant(0.0)

    # Find exact solution:
    u_exact = fe.Expression(
        "(1./2 - x[0]) * (2 * x[0] + eps/(2*pi) * sin(2*pi*x[0]/eps)) "
        "+ eps*eps/(2*pi*2*pi) * (1 - cos(2*pi*x[0]/eps)) + x[0]*x[0]",
        eps=eps, degree=2)

    def boundary(x, on_boundary):
        return on_boundary

    bc = fe.DirichletBC(V, u_D, boundary)

    # Define variational problem

    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    f = fe.Constant(1)
    A = fe.Expression('1./(2+cos(2*pi*x[0]/eps))', eps=eps, degree=2)
    a = A * fe.dot(fe.grad(u), fe.grad(v)) * fe.dx
    L = f * v * fe.dx

    # Compute solution
    u = fe.Function(V)
    fe.solve(a == L, u, bc)

    if plot:
        # Plot solution
        fe.plot(u)
        fe.plot(u_exact, mesh=mesh)
        # Hold plot
        fe.interactive()

    # # Save solution to file in VTK format
    # vtkfile = fe.File('poisson/solution.pvd')
    # vtkfile << u

    # Compute error
    err_norm = fe.errornorm(u_exact, u, 'L2')
    return err_norm


if __name__ == "__main__":
    h_list = [1. / 2 ** i for i in range(3, 11)]
    eps_list = [1. / 2 ** i for i in range(3, 15)]
    errors = np.zeros((len(h_list), len(eps_list)))
    for h_step, h in enumerate(h_list):
        for eps_step, eps in enumerate(eps_list):
            err = solve_poisson_eps(h, eps, plot=False)
            errors[h_step, eps_step] = err
    conv_rates = errors[:-1, :] / errors[1:, :]
    # Logplot of errors as function of grid size
    plt.semilogy(errors)
    plt.xlabel('h')
    plt.legend(['eps = %.2e' % eps for eps in eps_list], loc='best')
    plt.title("Logplot of errors for different epsilon")
    # Plot of convergence rates as function of grid size
    plt.figure()
    plt.plot(conv_rates)
    plt.xlabel('h')
    plt.title("plot of convergence rates for different epsilon")
    plt.legend(['eps = %.2e' % eps for eps in eps_list], loc='best')
    plt.show()

    # Make a plot that shows the order of convergence as a function of eps/h?
