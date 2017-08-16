#!/usr/bin/env python
"""
FEniCS script for solving -div(-A(x/eps)u) = f in 2/3D
Solves the cell problem, computes the homogenized diffusion coefficient A*
then solve the general equation with A*
"""

from __future__ import print_function
import time as time1
import numpy as np
from fenics import *
import fenics as fe
from dolfin import *
from mshr import *
import dolfin as do
import matplotlib.pyplot as plt


class PoissonSolver:
    def __init__(self, h, eps, dim, degree = 2, mesh=None, time=None):
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
        
    
            
        a_eps = '1./(2+cos(2*pi*x[0]/eps))'
        self.e_is = [fe.Constant(1.)]
        if self.dim == 2:
            if mesh is None:
                self.mesh = fe.UnitSquareMesh(self.n, self.n)
            else:
                self.mesh = mesh
                #fe.plot(self.mesh)
                #plt.show()
            a_eps = '1./(2+cos(2*pi*(x[0]+2*x[1])/eps))'
            self.e_is = [fe.Constant((1., 0.)), fe.Constant((0., 1.))]
            self.diff_coef = fe.Expression(a_eps, eps=self.eps, degree=2, domain=self.mesh)
        elif self.dim == 3:
            if mesh is None:
                self.mesh = fe.UnitCubeMesh(self.n, self.n, self.n)
            else:
                self.mesh = mesh
            a_eps = '(sin(2*pi*{})*x[0]*x[0]+1.2)/(2+cos(2*pi*(x[0]+2*x[1]+x[2])/eps))'.format(time)#(x[1]*x[1]+x[2]*x[2]+{}*x[0]*x[0])
            print(a_eps, self.eps)
            self.e_is = [fe.Constant((1., 0., 0.)), fe.Constant((0., 1., 0.)), fe.Constant((0., 0., 1.))]
            self.diff_coef = fe.Expression(a_eps, eps=self.eps, degree=2, domain=self.mesh)

        else:
            self.mesh = fe.UnitIntervalMesh(self.n)
            a_eps = '1./(2+cos(2*pi*x[0]/eps))'
            self.dim = 1
            #print("Solving rapid varying Poisson problem in R^%d" % self.dim)
            self.diff_coef = fe.Expression(a_eps, eps=self.eps, degree=2, domain=self.mesh)
        self.a_y = fe.Expression(a_eps.replace("/eps",""), degree=2, domain=self.mesh)
        self.function_space = fe.FunctionSpace(self.mesh, 'P', degree)
        self.solution = fe.Function(self.function_space)
        self.cell_solutions = [fe.Function(self.function_space) for _ in range(self.dim)]
        self.eff_diff = np.zeros((self.dim, self.dim))
        # Define boundary condition
        self.bc_function = fe.Constant(0.0)
        self.f = fe.Constant(0.001)
        
    


    def solve_exact(self, solve_method=None, prec=None):
        """
        Attempts to solve the PDE exactly for given epsilon > 0
        :return: None
        """
        self._solve_pde(self.diff_coef, solve_method, prec)

    def solve_homogenized(self):
        """
        Solves the limit (epsilon to 0) PDE
        :return: None
        """
        self._solve_cell_problems()
        eff_diff_coef = self._compute_effective_diffusion()
        #print("Computed effective diffusion coefficent:\n%s" % eff_diff_coef.values())
        self._solve_pde(eff_diff_coef)

    def _solve_pde(self, diff_coef, solve_method = None, prec=None):
        # Actual PDE solver for any coefficient diff_coef
        u = fe.TrialFunction(self.function_space)
        v = fe.TestFunction(self.function_space)
        a = fe.dot(diff_coef * fe.grad(u), fe.grad(v)) * fe.dx
        L = self.f * v * fe.dx
        bc = fe.DirichletBC(self.function_space, self.bc_function, PoissonSolver.boundary)
        if solve_method is not None:
            # Note: If solve_method is chosen, the solver will now be set up without
            # preconditioner unless otherwise stated
            problem = fe.LinearVariationalProblem(a, L, self.solution, bc)
            solver = fe.LinearVariationalSolver(problem)
            prm = fe.parameters["krylov_solver"] # short form
            #prm["absolute_tolerance"] = 1E-7
            #prm["relative_tolerance"] = 1E-4
            #print()rm["maximum_iterations"] = 1000
            solver.parameters["linear_solver"] = solve_method
            if prec is None:
                print('No preconditioner')
            solver.parameters["preconditioner"] = prec
            solver.solve()
        else:
            fe.solve(a==L,self.solution, bc)#self._solve(a, L, bc)

    def _solve(self,a, L, bc):
        #solver = KrylovSolver("gmres", "ml_amg")
        problem = fe.LinearVariationalProblem(a, L, self.solution, bc)
        solver = fe.LinearVariationalSolver(problem)
        prm = fe.parameters["krylov_solver"] # short form
        #prm["absolute_tolerance"] = 1E-7
        #prm["relative_tolerance"] = 1E-4
        #prm["maximum_iterations"] = 1000
        solver.parameters["linear_solver"] = "gmres"
        #solver.parameters["krylov_solver"]["relative_tolerance"] = 5e-4
        #solver.parameters["krylov_solver"]["monitor_convergence"] = True
       # solver.parameters["krylov_solver"]["maximum_iterations"] = 1000
        solver.solve()


    def _solve_cell_problems(self):
        # Solves the cell problems (one for each space dimension)
        w = fe.TrialFunction(self.function_space)
        v = fe.TestFunction(self.function_space)
        a = self.a_y * fe.dot(fe.grad(w), fe.grad(v)) * fe.dx
        for i in range(self.dim):
            L = fe.div(self.a_y * self.e_is[i]) * v * fe.dx
            bc = fe.DirichletBC(self.function_space, self.bc_function, PoissonSolver.boundary)
            fe.solve(a == L, self.cell_solutions[i], bc,solver_parameters=dict(linear_solver='cg',
                             preconditioner='ilu'))
            
            
    def _compute_effective_diffusion(self):
        # Uses the solutions of the cell problems to compute the harmonic average of the diffusivity
        for i, j in np.ndindex((self.dim, self.dim)):
            integrand_ij = fe.project(
                self.a_y*fe.dot(self.e_is[i] + fe.grad(self.cell_solutions[i]),
                                self.e_is[j] + fe.grad(self.cell_solutions[j])), self.function_space)
            self.eff_diff[i, j] = fe.assemble(integrand_ij * fe.dx)
        
        return fe.Constant(self.eff_diff)

    def store_solution(self, file_handle=None, time=None):
        # Store the solution to file
        if file_handle is None:
            file_handle = fe.File('results/solution.pvd')
        
        
        if time is None:
            file_handle << self.solution
        else:
            print(float(time))
            u = self.solution.copy()
            u.rename('heat', '1222')
            file_handle << (u, float(time))

    def plot(self):
        # Quickplot of the solution
        plt.figure()
        fe.plot(self.diff_coef, mesh=self.mesh)
        plt.figure()
        fe.plot(self.solution)
        fe.interactive()
        

    @staticmethod
    def boundary(x, on_boundary):
        # FeNiCS boundary method
        return on_boundary


def compare_cars(eps_list,n_cells, file_name, dim=2):
    
    errors = np.zeros(len(eps_list))
    # After running compare_solutions_cell_problem, we know that
    # h= 1/2^7 should be sufficient in all cases for the cell problem.
    # As the cell problem does not depend on epsilon, we only solve
    # it once.
    s_cell = PoissonSolver(1/2**9, 42, dim)
    s_cell._solve_cell_problems()
    eff_diff_coef = s_cell._compute_effective_diffusion()

    #car_mesh = create_car_mesh(30)
    fine_car_mesh = create_car_mesh(n_cells)
    s_homo = PoissonSolver(1/n_cells, 42, dim, degree=2, mesh=fine_car_mesh)
    s_homo._solve_pde(eff_diff_coef)
    fn = file_name+'_homogenized.pvd'
    s_homo.store_solution(fn)
    plot(fine_car_mesh)
    plt.figure()
    for eps_step, eps in enumerate(eps_list):
        
        # Compute reference solution for this epsilon
        s_ref = PoissonSolver(1/n_cells, eps, dim, degree =5, mesh=fine_car_mesh)
        s_ref.solve_exact()
        fe.plot(s_ref.solution)
        plt.axis('off')
        plt.title('Reference solution, eps = %.2e \n \n \n \n ' % eps)
        plt.savefig('../report/images/carw_reference_eps_power_%i.png' % eps_step)
        plt.figure()
        fn = file_name+'%f.pvd' % eps
        print(fn)
        s_ref.store_solution(fn)

        err_norm = fe.errornorm(s_ref.solution, s_homo.solution, 'L2')
        errors[eps_step] = err_norm

    
    fe.plot(s_homo.solution)
    plt.title('Homogenized solution \n \n \n \n ')
    plt.axis('off')
    plt.savefig('../report/images/carw_homogenized.png')
    fe.interactive()
    plt.show(block=False)
    return(errors)

def compare_solutions_cell_problem(eps_list, h_cell_list, dim=2):
    
    errors = np.zeros((len(h_cell_list), len(eps_list)))
    for eps_step, eps in enumerate(eps_list):
        # Compute reference solution for this epsilon
        s_ref = PoissonSolver(1/2**7, eps, dim, degree =4)
        s_ref.solve_exact()

        for h_step, h in enumerate(h_cell_list):
        
            # Compute effective diffusion coefficent with different
            # cell problem discretizations
            s_cell = PoissonSolver(h,eps,dim)
            s_cell._solve_cell_problems()
            eff_diff_coef = s_cell._compute_effective_diffusion()

            # Solve homogenized on fine mesh (i.e., the error from
            # this step is negligible compared to that of the
            # eff_diff_coef computation)
            s_homo = PoissonSolver(1/2**7, 42, dim)
            s_homo._solve_pde(eff_diff_coef)
            # Compute error
            err_norm = fe.errornorm(s_ref.solution, s_homo.solution, 'L2')
            errors[h_step, eps_step] = err_norm


    print(errors)
    return(errors)


def compare_solutions_global(eps_list, h_global_list, dim=2):
    
    errors = np.zeros((len(h_global_list), len(eps_list)))
    # After running compare_solutions_cell_problem, we know that
    # h= 1/2^7 should be sufficient in all cases for the cell problem.
    # As the cell problem does not depend on epsilon, we only solve
    # it once.
    s_cell = PoissonSolver(1/2**7, 42, dim, degree=2)
    s_cell._solve_cell_problems()
    eff_diff_coef = s_cell._compute_effective_diffusion()
    
    for eps_step, eps in enumerate(eps_list):
        # Compute reference solution for this epsilon
        s_ref = PoissonSolver(1/2**7, eps, dim, degree =4)
        s_ref.solve_exact()        
        
        for h_step, h in enumerate(h_global_list):
        
            # Solve homogenized on fine mesh (i.e., the error from
            # this step is negligible compared to that of the
            # eff_diff_coef computation)
            s_homo = PoissonSolver(h, eps, dim)
            s_homo._solve_pde(eff_diff_coef)
            # Compute error
            err_norm = fe.errornorm(s_ref.solution, s_homo.solution, 'L2')
            errors[h_step, eps_step] = err_norm
    print(errors)
    return(errors)

#---------------------------------------------------------------------#


def compare_solvers(eps, h_global, solver_types, prec_list):
    dim = 2
    times_ref = np.zeros((len(prec_list),len(solver_types)))
    times_homo = np.zeros((len(prec_list),len(solver_types)))
    s_cell = PoissonSolver(1/2**7, 42, dim)
    s_cell._solve_cell_problems()
    eff_diff_coef = s_cell._compute_effective_diffusion()
    


    for solver_step, solver in enumerate(solver_types):
        for prec_step, prec in enumerate(prec_list):
            
            h = h_global
            print(solver, prec)
        
            s_ref = PoissonSolver(h, eps, dim, degree =1)
            try:
                t0 = time1.time()
                s_ref.solve_exact(solver, prec)        
                t_ref = time1.time()-t0
            except Exception:
                t_ref = None
        
            # Solve homogenized on fine mesh (i.e., the error from
            # this step is negligible compared to that of the
            # eff_diff_coef computation)
            s_homo = PoissonSolver(h, eps, dim, degree = 1)
            try:
                t0 = time1.time()
                s_homo._solve_pde(eff_diff_coef, solver, prec)
                t_homo = time1.time()-t0
            except Exception:
                t_homo = None
        
        
            times_ref[prec_step, solver_step] = t_ref
            times_homo[prec_step, solver_step] = t_homo
    return times_ref, times_homo

#---------------------------------------------------------------------#
   
    
def plot_errors(e_t, c_r, epsilons,h_val, problem='global'):
    if problem =='cell':
        xl = '$h_{cell}$'
    else:
        xl = '$h_{global}$'

    plt.figure()
    plt.semilogy(e_t)
    plt.xlabel(xl)
    plt.ylabel('e')
    plt.title("Logplot of errors for different epsilon")
    plt.legend(['eps = %.2e' % eps for eps in epsilons], loc='best')
    plt.xticks(np.arange(len(h_val)), h_val, rotation='vertical')
    
    # Plot of convergence rates as function of grid size
    plt.figure()
    plt.plot(c_r)
    plt.ylabel('rate')
    plt.xlabel(xl)
    plt.title("Plot of convergence rates for different epsilon")
    plt.xticks(np.arange(len(h_val)), h_val, rotation='vertical')
    plt.legend(['eps = %.2e' % eps for eps in eps_list], loc='best')
    plt.show(block=False)

#---------------------------------------------------------------------#

def mesh_3d(mesh_size):

    box = Box(Point(0, 0, 0), Point(1, 1, 1))
    sphere1 = Sphere(Point(-0.1,0, 1), 0.7)
    sphere2 = Sphere(Point(.1, .2, 1), 0.4)
    cone = Cone(Point(0, 0, 1), Point(0, 0, -1.5), 1., segments=20)
    g3d = sphere1+ cone #- sphere

    # Test printing
    #info("\nCompact output of 3D geometry:")
    #info(g3d)
    #info("\nVerbose output of 3D geometry:")
    #info(g3d, True)

    # Plot geometry
    #plot(g3d, "3D geometry (surface)")

    # Generate and plot mesh
    mesh3 = generate_mesh(g3d, mesh_size)
    info(mesh3)
    plot(mesh3, "3D mesh")
    plt.axis('equal')
    interactive()
    return mesh3

#---------------------------------------------------------------------#
def create_car_mesh(mesh_size):
    '''
    Creates and returns a dolfin mesh of a cartoon car. 
    mesh_size: int, 30-40 gives reasonably good meshes of approx.
    1000-2000 cells.
    '''
    
    # Create list of polygonal domain vertices
    bodywork = [Point(0.0, 0.3),
                Point(3, 0.3),
                Point(2.95, 0.7),
                Point(2.6, 0.8),
                Point(2.3, 1.3),
                Point(1.0, 1.3),
                Point(0.65, 0.8),
                Point(0.05, 0.7),
                Point(0.0, 0.3)]

    window1  = [Point(0.8, 0.8),
                Point(1.45, 0.8),
                Point(1.45, 1.2),
                Point(1.05, 1.2),
                Point(0.8, 0.8)]
    window2  = [Point(1.65, 0.8),
                Point(2.3, 0.8),
                Point(2.1, 1.2),
                Point(1.65, 1.2),
                Point(1.65, 0.8)]

    

 
    bw = Polygon(bodywork)
    window1 = Polygon(window1)
    window2 = Polygon(window2)
    wheel1 = Circle(Point(2.25,0.27), .3)
    wheel2 = Circle(Point(0.75,0.27), .3)
    car = bw+ wheel1+wheel2-window1-window2
    # Generate mesh and plot
    mesh = generate_mesh(car, mesh_size)
    print('''Created mesh with %i cells of maximum size %f 
        (compare to $\epsilon$ for reference solutions)'''
          %(mesh.cells().shape[0], mesh.hmax()))
    plot(mesh)
    plt.show(block=False)
    input('Hit return to close plot')
    plt.close()
    return mesh

#---------------------------------------------------------------------#

def evaluate_timestep(s_global, t, h_cell):
    # Cell
    d = 3
    s_cell = PoissonSolver(h_cell, 1, d, time = t)
    
    s_cell._solve_cell_problems()
    eff_diff_coef = s_cell._compute_effective_diffusion()
    print('A_eps for time step {} is {}'.format(t, eff_diff_coef.values()) )
    # Global
    s_global._solve_pde(eff_diff_coef)


    
def plot_solver_data(times_homo, times_ref,h_gl, solver_types, preconditioners):
    plt.close('all')
    plt.yscale('log')
    
    dim = len(times_ref[0])
    w = 0.75
    dimw = w / dim
    x = np.arange(times_ref.shape[1])
    for i in range(times_ref.shape[0]):
        y = [d[i] for d in times_ref]
        b = plt.bar(x + i * dimw, y, dimw, bottom=0.001)
    plt.legend(solver_types)
    plt.xticks(np.arange(len(preconditioners))+np.ones(len(preconditioners))*dimw/2, preconditioners, rotation='vertical')
    
    axes = plt.gca()
    axes.set_ylim([.3,24])
    plt.ylabel('time', rotation='vertical')
    nc = 2**(2*h_gl)
    plt.title('Solution times for full problem, {:.1E} cells'.format(nc))#% h_global_list[0])

    plt.figure()
    plt.yscale('log')
    for i in range(times_homo.shape[0]):
        y = [d[i] for d in times_homo]
        b = plt.bar(x + i * dimw, y, dimw, bottom=0.001)
    plt.legend(solver_types)
    plt.xticks(np.arange(len(preconditioners))+np.ones(len(preconditioners))*dimw/2, preconditioners, rotation='vertical')
    
    axes = plt.gca()
    axes.set_ylim([.3,24])
    plt.ylabel('time', rotation='vertical')
    
    plt.title('Solution times for homogenized problem, {:.1E} cells'.format(nc))#% h_global_list[0])
    
    plt.show(block=False)
    return
    plt.semilogy(times_ref)
    plt.title('Solution times for full problem, h = %f' % h_global_list[0])
    plt.legend(solver_types)
    plt.xticks(np.arange(len(preconditioners)), preconditioners, rotation='vertical')
    
    plt.figure()
    plt.semilogy(times_homo)
    plt.xticks(np.arange(len(preconditioners)), preconditioners, rotation='vertical')
    plt.title('Solution times for homogenized problem, h = %f' % h_global_list[0])
    plt.legend(solver_types)
    

#---------------------------------------------------------------------#
    
def stored_solutions():
    """
    For runs which take a while...
    """
    #solver_comparison, h_global = 9, eps = 0
    times_ref = np.array(
        [[  8.191715,            float('nan'),  13.05460715,   8.19411731],
         [  5.11841059,  36.65315652,   7.35329151,   5.84721565],
         [  2.63757253,  12.59475803,   4.49874902,   2.95621014],
         [  6.91874003,          float('nan'),  14.43135071,   6.99962544]])
    times_homo = np.array(
        [[  6.21782589,          float('nan'),  11.81072927,   6.34943771],
         [  4.85342669,  41.89151621,   8.58132935,   6.738096  ],
         [  2.293957,    10.9519248,    4.49466801,   2.3979404 ],
         [  6.51827908,          float('nan'),  11.3594017,    7.83625436]])
    tr_8 = np.array([[  0.99127626,  23.17382479,   1.28662777,   1.04913497],
       [  0.69222307,   3.23113894,   0.83439064,   0.99728251],
       [  0.62268639,   0.87030935,   0.51625276,   0.42412806],
       [  0.82240033,  12.16858339,   1.14180565,   0.79825974]])
    th_8 = np.array([[  0.84229136,  14.1169703 ,   1.03939867,   1.08646965],
       [  1.03686857,   2.92545295,   1.24408674,   1.00220394],
       [  0.44128227,   0.71016026,   0.43579865,   0.39884138],
       [  1.69220471,  15.88684011,   2.1891706 ,   0.84371662]])
    return tr_8, th_8
#---------------------------------------------------------------------#


if __name__ == "__main__":
    # The script may be used for three different purposes 
    solver_comparison = False
    cell_and_global = False
    geometry = False
    show_off = True
    if solver_comparison:
        n_h = 1
        n_e = 1
        epsilon = 3
        n_glob_h = 1
        h0=0
        h_global = 8#3 #7,9 prøv ni
        
        hglob0 = h_global
        print('ncells will be ',2**(hglob0**2))
        fe.list_krylov_solver_methods()
        fe.list_krylov_solver_preconditioners()
        
        solver_types = ['cg', 'gmres', 'minres', 'bicgstab'] 
        preconditioners = ['none', 'sor', 'ilu', 'jacobi']
        # problems with cg and minres for icc preconditioner, change to true to get the corresponding plots
        do_icc = False
        if do_icc:
            solver_types = [solver_types[1],solver_types[3]] #The two which work with icc
            preconditioners = ['icc', 'none']
        h_cell_list = [1. / 2 ** i for i in range(h0, h0+n_h)]
        h_global_list = [1. / 2 ** i for i in range(hglob0, hglob0+n_glob_h)]
        
        
        #times_ref, times_homo=compare_solvers(epsilon, 1/2**h_global, solver_types, preconditioners)
        times_ref, times_homo= stored_solutions()
        
        print(repr(times_ref))
        print(repr(times_homo))
        plot_solver_data(times_homo, times_ref,h_global, solver_types, preconditioners)

    if cell_and_global:    
    
        n_h = 7
        n_e = 4
        eps0 = 0
        n_glob_h = 7
        
        h0=0
        hglob0 = 0
        h_cell_list = [1. / 2 ** i for i in range(h0, h0+n_h)]
        h_global_list = [1. / 2 ** i for i in range(hglob0, hglob0+n_glob_h)]
        eps_list = [1. / 2 ** i for i in range(eps0, eps0+n_e)]
        plt.close('all')
        
        t = time1.time()
        if n_h > 0:
            error_table = compare_solutions_cell_problem(eps_list,h_cell_list)
        if n_glob_h > 0:
            error_table_global = compare_solutions_global(eps_list,h_global_list)
        print('error evaluation took: ', time1.time()-t)

        if n_h > 1:
            conv_rates = error_table[:-1, :] / error_table[1:, :]
            plot_errors(error_table, conv_rates,eps_list,h_cell_list, problem = 'cell')
        if n_glob_h > 1:
            conv_rates_global = error_table_global[:-1, :] / error_table_global[1:, :]
            plot_errors(error_table_global, conv_rates_global,eps_list,h_global_list,problem = 'global')



    if geometry:
        # Set parameters
        n_h = 1
        n_e = 4
        eps0 = 0
        n_cells = 300#7
        
        h0=0
        hglob0 = 0
        h_cell_list = [1. / 2 ** i for i in range(h0, h0+n_h)]
        eps_list = [1. / 2 ** i for i in range(eps0, eps0+n_e)]
        plt.close('all')
        file_name= 'results/car_solution'

        # Solve problem
        t = time1.time()
        #create_car_mesh(30)
        #errors = compare_cars(eps_list, n_cells, file_name)
        print('Full evaluation took ', time1.time()-t)
        #errors = [  1.62851640e-05,   8.26573364e-06,   4.87418268e-06,   3.60014481e-06] # without windows
        errors = [  8.71610346e-06,   4.52455804e-06,   2.28755036e-06,   1.33245985e-06]
        print('errors: ', errors)
        # Plot convergence:
        print('eps',eps_list)
        plt.figure()
        plt.semilogy(errors)
        plt.xlabel('$\epsilon$')
        plt.ylabel('e')
        plt.title("Logplot of errors")
        plt.xticks(np.arange(len(eps_list)), eps_list, rotation='vertical')
        
        


    


    if show_off:
        # Set parameters
        dim = 3
        mesh_size = 25#35
        h_cell = 1e-1
        epsilon = 1/2**0
        file_name= 'results/3d_solution.pvd'
        f = fe.File(file_name)
        # Make mesh
        mesh_3= mesh_3d(mesh_size)
        # Initalize
        
        s_global = PoissonSolver(1/mesh_size, epsilon, dim, mesh=mesh_3, time=0)
        #plot(s_global, mesh=self.mesh)
        #plot.show()
        # Loop in time
        n_t=20
        
        timesteps = np.linspace(0,3,n_t)
        for t_step, t in enumerate(timesteps):
            print('Solve at t = ', t)
            evaluate_timestep(s_global, t, h_cell)
            print('Store for time step {} of {}'.format(t_step, n_t))
            s_global.store_solution(f, t)
            s_global.plot()
    #f << s_global.solution
    #plt.show(block=False)
    input('Hit enter to close exit (and close plots)')
