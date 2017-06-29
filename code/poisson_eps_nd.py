#!/usr/bin/env python
"""
FEniCS code for -div(-A(x/eps)u) = f in 2/3D
Homogenized solution approximation for fixed eps > 0
"""

from __future__ import print_function
import numpy as np
import fenics as fe
import matplotlib.pyplot as plt


class PoissonSolver:
    def __init__(self, h, eps, dim):
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
            a_eps = '1./(2+cos(2*pi*(x[0]+3*x[1]+x[2]/3)/eps))'
            self.e_is = [fe.Constant((1., 0., 0.)), fe.Constant((0., 1., 0.)), fe.Constant((0., 0., 1.))]
        else:
            self.dim = 1
        print("Solving rapid varying Poisson problem in R^%d" % self.dim)
        self.diff_coef = fe.Expression(a_eps, eps=self.eps, degree=2, domain=self.mesh)
        self.function_space = fe.FunctionSpace(self.mesh, 'P', 2)
        self.solution = fe.Function(self.function_space)
        self.cell_solutions = [fe.Function(self.function_space) for _ in range(self.dim+1)]
        self.eff_diff = np.zeros((self.dim, self.dim))
        # Define boundary condition
        self.bc_function = fe.Constant(0.0)

        self.f = fe.Constant(1)

    def solve_exact(self):
        u = fe.TrialFunction(self.function_space)
        v = fe.TestFunction(self.function_space)
        a = self.diff_coef * fe.dot(fe.grad(u), fe.grad(v)) * fe.dx
        L = self.f * v * fe.dx
        bc = fe.DirichletBC(self.function_space, self.bc_function, PoissonSolver.boundary)

        fe.solve(a == L, self.solution, bc)

    def solve_cell_problems(self):
        for i in range(self.dim):
            w = fe.TrialFunction(self.function_space)
            v = fe.TestFunction(self.function_space)
            a = self.diff_coef * fe.dot(fe.grad(w), fe.grad(v)) * fe.dx
            L = fe.div(self.diff_coef * self.e_is[i]) * v * fe.dx  # move these lines outside of loop
            bc = fe.DirichletBC(self.function_space, self.bc_function, PoissonSolver.boundary)

            fe.solve(a == L, self.cell_solutions[i], bc)
            fe.plot(self.cell_solutions[i])

    def compute_effective_diffusion(self):
        for i, j in np.ndindex((self.dim, self.dim)):
            integrand_ij = fe.project(
                fe.dot(self.e_is[i] + fe.grad(self.cell_solutions[i]),
                       self.e_is[j] + fe.grad(self.cell_solutions[j])), self.function_space)
            self.eff_diff[i, j] = fe.assemble(integrand_ij * fe.dx)
        print(self.eff_diff)

    def solve_homogenized_problem(self):
        u = fe.TrialFunction(self.function_space)
        v = fe.TestFunction(self.function_space)
        a = fe.dot(fe.Constant(self.eff_diff) * fe.grad(u), fe.grad(v)) * fe.dx
        L = self.f * v * fe.dx
        bc = fe.DirichletBC(self.function_space, self.bc_function, PoissonSolver.boundary)
        fe.solve(a == L, self.solution, bc)

    def store_solution(self):
        vtkfile = fe.File('results/solution.pvd')
        vtkfile << self.solution

    def plot(self):
        fe.plot(self.diff_coef, mesh=self.mesh)
        fe.plot(self.solution)
        fe.interactive()

    @staticmethod
    def boundary(x, on_boundary):
        return on_boundary


if __name__ == "__main__":
    h = 1. / 2 ** 5
    eps = 1. / 2 ** 2
    dim = 2
    solver = PoissonSolver(h, eps, dim)
    solver.solve_cell_problems()
    solver.compute_effective_diffusion()
    solver.solve_homogenized_problem()
    solver.plot()
