#!/usr/bin/env python
"""
FEniCS script for solving -div(-A(x/eps)u) = f in 2/3D
Solves the cell problem, computes the homogenized diffusion coefficient A*
then solve the general equation with A*
"""

from __future__ import print_function
import numpy as np
import fenics as fe
import matplotlib.pyplot as plt


class PoissonSolver:
    def __init__(self, h, eps, dim):
        """
        Initializing poisson solver
        :param h: mesh size (of unit interval discretisation)
        :param eps: small parameter
        :param dim: dimension (in {1,2,3})
        """
        self.h = h
        self.n = int(1 / h)
        self.eps = eps
        self.dim = dim
        self.mesh = fe.UnitIntervalMesh(self.n)
        a_eps = '1./(2+cos(2*pi*x[0]/eps))'
        self.e_is = [fe.Constant(1.)]
        if self.dim == 2:
            self.mesh = fe.UnitSquareMesh(self.n, self.n)
            a_eps = '1./(2+cos(2*pi*(x[0]+2*x[1])/eps))'
            self.e_is = [fe.Constant((1., 0.)), fe.Constant((0., 1.))]
        elif self.dim == 3:
            self.mesh = fe.UnitCubeMesh(self.n, self.n, self.n)
            a_eps = '1./(2+cos(2*pi*(x[0]+3*x[1]+6*x[2])/eps))'
            self.e_is = [fe.Constant((1., 0., 0.)), fe.Constant((0., 1., 0.)), fe.Constant((0., 0., 1.))]
        else:
            self.dim = 1
        print("Solving rapid varying Poisson problem in R^%d" % self.dim)
        self.diff_coef = fe.Expression(a_eps, eps=self.eps, degree=2, domain=self.mesh)
        self.a_y = fe.Expression(a_eps.replace("/eps",""), degree=2, domain=self.mesh)
        self.function_space = fe.FunctionSpace(self.mesh, 'P', 2)
        self.solution = fe.Function(self.function_space)
        self.cell_solutions = [fe.Function(self.function_space) for _ in range(self.dim)]
        self.eff_diff = np.zeros((self.dim, self.dim))
        # Define boundary condition
        self.bc_function = fe.Constant(0.0)

        self.f = fe.Constant(1)

    def solve_exact(self):
        """
        Attempts to solve the PDE exactly for given epsilon > 0
        :return: None
        """
        self._solve_pde(self.diff_coef)

    def solve_homogenized(self):
        """
        Solves the limit (epsilon to 0) PDE
        :return: None
        """
        self._solve_cell_problems()
        eff_diff_coef = self._compute_effective_diffusion()
        print("Computed effective diffusion coefficent:\n%s" % eff_diff_coef.values())
        self._solve_pde(eff_diff_coef)

    def _solve_pde(self, diff_coef):
        # Actual PDE solver for any coefficient diff_coef
        u = fe.TrialFunction(self.function_space)
        v = fe.TestFunction(self.function_space)
        a = fe.dot(diff_coef * fe.grad(u), fe.grad(v)) * fe.dx
        L = self.f * v * fe.dx
        bc = fe.DirichletBC(self.function_space, self.bc_function, PoissonSolver.boundary)
        fe.solve(a == L, self.solution, bc)

    def _solve_cell_problems(self):
        # Solves the cell problems (one for each space dimension)
        w = fe.TrialFunction(self.function_space)
        v = fe.TestFunction(self.function_space)
        a = self.a_y * fe.dot(fe.grad(w), fe.grad(v)) * fe.dx
        for i in range(self.dim):
            L = fe.div(self.a_y * self.e_is[i]) * v * fe.dx
            bc = fe.DirichletBC(self.function_space, self.bc_function, PoissonSolver.boundary)
            fe.solve(a == L, self.cell_solutions[i], bc)
            fe.plot(self.cell_solutions[i])

    def _compute_effective_diffusion(self):
        # Uses the solutions of the cell problems to compute the harmonic average of the diffusivity
        for i, j in np.ndindex((self.dim, self.dim)):
            integrand_ij = fe.project(
                self.a_y*fe.dot(self.e_is[i] + fe.grad(self.cell_solutions[i]),
                                self.e_is[j] + fe.grad(self.cell_solutions[j])), self.function_space)
            self.eff_diff[i, j] = fe.assemble(integrand_ij * fe.dx)
        return fe.Constant(self.eff_diff)

    def store_solution(self):
        # Store the solution to file
        vtkfile = fe.File('results/solution.pvd')
        vtkfile << self.solution

    def plot(self):
        # Quickplot of the solution
        fe.plot(self.diff_coef, mesh=self.mesh)
        fe.plot(self.solution)
        fe.interactive()

    @staticmethod
    def boundary(x, on_boundary):
        # FeNiCS boundary method
        return on_boundary


if __name__ == "__main__":
    h = 1. / 2 ** 5
    eps = 1. / 2 ** 2
    dim = 2
    solver = PoissonSolver(h, eps, dim)
    solver.solve_homogenized()
    solver.store_solution()
    solver.plot()
