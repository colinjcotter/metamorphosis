from firedrake import *
from firedrake_adjoint import *

nx = 40
Length = 1.0
Height = 1.0
ny = 40
mesh = PeriodicRectangleMesh(nx, ny, Length, Height,
                             direction="x",
                             quadrilateral=True)

VI = FunctionSpace(mesh, "DG", 1)
VZ = FunctionSpace(mesh, "DG", 3)
T = FunctionSpace(mesh, "HDiv Trace", 1)
U = FunctionSpace(mesh, "CG", 1)

hybrid = True

#building funny quadrature
cell = U.finat_element.cell
subcell = cell.construct_subelement(cell.get_dimension() - 1)
verts = subcell.get_vertices()
assert(len(verts)==2)
assert(len(verts[0])==1)
from finat import quadrature, point_set
points = point_set.PointSet(verts)
weights = [0.5, 0.5]
Vrule = quadrature.QuadratureRule(points, weights)

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
        'Gradient Tolerance': 1e-3,
        'Step Tolerance': 1e-16,
        'Relative Step Tolerance': 1e-10,
    'Iteration Limit': 5
    }
}

if hybrid:
    W = VZ * VI * T
else:
    W = VI * VZ * T

w = Function(W)
if hybrid:
    z, I, theta = split(w)
    dz, dI, dtheta = TestFunctions(W)
else:
    I, z, theta = split(w)
    dI, dz, dtheta = TestFunctions(W)

eps = Constant(1.0e-4)
c = Constant(0.2)
sig = Constant(1.0e-2)

u = Function(U).assign(0.2)

n = FacetNormal(mesh)

x, t = SpatialCoordinate(mesh)

t0 = Constant(0.02)
H = lambda x: 0.5*(tanh(-x/t0) + 1)

I0 = H(abs(x-0.5 - c*t) - 0.2)
#I0 = conditional(abs(x-0.5 - c*t) < 0.2, 1.0, 0.0)
#I0 = conditional(abs(x-0.5) < 0.2, 1.0, 0.0)
#u = Function(U).interpolate(c)
b = as_vector([u,1])

thetaS = theta('+')
dthetaS = dtheta('+')

dS0 = dS(degree=6)
ds0 = ds(degree=6)

eqn = (
    dz*z*dx
    + div(b*dz)*I*dx
    #- div(b)*dz*I*dx density mode
    - jump(b*dz,n)*thetaS*dS - inner(b*dz,n)*I0*ds
    +  dI*div(b*z)*dx
    #- div(b)*z*dI*dx density mode
    + jump(b*z,n)*dthetaS*dS
)
eqn += sig*dtheta*theta*ds(rule=Vrule) + sig*dthetaS*thetaS*dS(rule=Vrule)

if hybrid:
    sparams = {'ksp_type':'gmres',
               'ksp_converged_reason':None,
               'mat_type':'matfree',
               'pmat_type':'matfree',
               'pc_type':'python',
               'pc_python_type':'firedrake.SCPC',
               'pc_sc_eliminate_fields': '0, 1',
               'condensed_field': {'ksp_type': 'preonly',
                                   'pc_type': 'lu',
                                   'pc_factor_mat_solver_type': 'mumps'}}

else:
    sparams = {'ksp_type':'preonly',
               'snes_monitor':None,
               'snes_linesearch':'basic',
               'mat_type':'aij',
               'pc_type':'lu',
               'pc_factor_mat_solver_type':'mumps'}

z0Prob = NonlinearVariationalProblem(eqn, w)
z0Solver = NonlinearVariationalSolver(z0Prob,
                                      solver_parameters=sparams)

du = TestFunction(U)
v0 = Function(U)
v1 = Function(U)
v2 = Function(U)

alpha = Constant(0.1)

v0eqn = (du*v1 + alpha**2*inner(du.dx(0),v1.dx(0)) - du*v0)*dx
v1eqn = (du*v2 + alpha**2*inner(du.dx(0),v2.dx(0)) - du*v1)*dx
v2eqn = (du*u + alpha**2*inner(du.dx(0),u.dx(0)) - du*v2)*dx

v0Prob = NonlinearVariationalProblem(v0eqn, v1)
v1Prob = NonlinearVariationalProblem(v1eqn, v2)
v2Prob = NonlinearVariationalProblem(v2eqn, u)

luparams = {'ksp_type':'preonly',
            'pc_factor_mat_solver_type':'mumps',
            'pc_type':'lu'}

v0Solver = NonlinearVariationalSolver(v0Prob,
                                        solver_parameters=luparams)
v1Solver = NonlinearVariationalSolver(v1Prob,
                                        solver_parameters=luparams)
v2Solver = NonlinearVariationalSolver(v2Prob,
                                        solver_parameters=luparams)

v0Solver.solve()
v1Solver.solve()
v2Solver.solve()
z0Solver.solve()
#zSolver.solve()

cu = Constant(0.001)

J = assemble(1./2*z*z*dx + cu*1./2*v0*u*dx)
m = Control(v0)
Jhat = ReducedFunctional(J, m)

# Define optimisation problem
problem = MinimizationProblem(Jhat)

solver = ROLSolver(problem, params_dict, inner_product="L2")
z_opt = solver.solve()

v0.assign(z_opt)
v0Solver.solve()
v1Solver.solve()
v2Solver.solve()
z0Solver.solve()
#zSolver.solve()

if hybrid:
    I, z, T = w.split()
else:
    z, I, T = w.split()
f = File('IZL2.pvd')

z0Solver.solve()
f.write(I,z,u)
