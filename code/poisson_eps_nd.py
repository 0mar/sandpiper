#!/usr/bin/env python
"""
FEniCS code for -div(-A(x/eps)u) = f in 1D
Unhomogenized solution approximation for fixed eps > 0
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
        if self.dim == 2:
            self.mesh = fe.UnitSquareMesh(self.n, self.n)
            a_eps = '1./(2+cos(2*pi*(x[0]+x[1])/eps))'
        if self.dim == 3:
            self.mesh = fe.UnitCubeMesh(self.n, self.n, self.n)
            a_eps = '1./(2+cos(2*pi*(x[0]+x[1])/eps))'
        else:
            self.dim = 1
        print("Solving rapid varying Poisson problem in R^%d" % self.dim)
        self.diff_coef = fe.Expression(a_eps, eps=eps, degree=2)
        self.function_space = fe.FunctionSpace(self.mesh, 'P', 1)
        self.solution = fe.Function(self.function_space)
        self.cell_solutions = [fe.Function(self.function_space) for _ in range(1, self.dim)]

        # Define boundary condition
        self.bc_function = fe.Constant(0.0)

        self.f = fe.Constant(1)

    def solve_exact(self):
        u = fe.TrialFunction(self.function_space)
        v = fe.TestFunction(self.function_space)
        a = self.diff_coef * fe.dot(fe.grad(u), fe.grad(v)) * fe.dx
        L = self.diff_coef * v * fe.dx
        bc = fe.DirichletBC(self.function_space, self.bc_function, PoissonSolver.boundary)

        fe.solve(a == L, self.solution, bc)

    def solve_cell_problems(self):
        pass

    def solve_homogenized_problem(self):
        pass

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
    eps = 1. / 2 ** 3
    dim = 2
    solver = PoissonSolver(h, eps, dim)
    solver.solve_exact()
    solver.plot()
