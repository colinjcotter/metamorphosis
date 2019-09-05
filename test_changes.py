#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/firedrake/lib/python3.6/site-packages/roltrilinos/lib
from firedrake import *
from firedrake_adjoint import *

nx = 150
Length = 1.0
Height = 1.0
ny = 150

mesh = PeriodicRectangleMesh(nx, ny, Length, Height,
                             direction="x",
                             quadrilateral=True)

V = FunctionSpace(mesh, "CG", 1)

cvals = [1.0, 2.0]
file9 = File('v.pvd')

vout = Function(V)

v = Function(V)
c = Function(V).assign(1.0)

u0 = Function(V)

u = TrialFunction(V)
w = TestFunction(V)

a = u*w*dx + c*inner(grad(u), grad(w))*dx
L = v*w*dx

x, y = SpatialCoordinate(mesh)

f = exp(sin(pi*x)*sin(pi*y))

params = {'ksp_type':'preonly',
          'pc_type':'lu'}

solve(a==L, u0, solver_parameters=params)

J = assemble(0.5*v*v*dx + 0.5*(u0-f)*(u0-f)*dx)

control = Control(v)
c_ctrl = Control(c)
rf = ReducedFunctional(J, control)
rf.optimize_tape()

# Define optimisation problem
problem = MinimizationProblem(rf)

# Now configure the optimisation solver
params_dict = {
    'General': {
        'Secant': {
            'Type': 'Limited-Memory BFGS',
            'Maximum Storage': 5
        }
    },
    'Step': {
        'Type': 'Line Search',
        'Line Search': {
            'Descent Method': {
                'Type': 'Quasi-Newton Method'
            },
            'Curvature Condition': {
                'Type': 'Strong Wolfe Conditions'
            }
        }
    },
    'Status Test': {
        'Gradient Tolerance': 1e-5,
        'Step Tolerance': 1e-16,
        'Relative Step Tolerance': 1e-10,
        'Iteration Limit': 100
    }
}

solver = ROLSolver(problem, params_dict, inner_product="L2")
for cval in cvals:
    c_ctrl.tape_value().assign(cval)
    z_opt = solver.solve()
    vout.assign(z_opt)
    file9.write(vout)
