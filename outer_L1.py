from firedrake import *
from firedrake_adjoint import *

file0 = File('IZ_l1.pvd')

nx = 40
Length = 1.0
Height = 1.0
ny = 40
mesh = PeriodicRectangleMesh(nx, ny, Length, Height,
                             direction="x",
                             quadrilateral=False)

VI = FunctionSpace(mesh, "DG", 1)
VZ = FunctionSpace(mesh, "DG", 3)
T = FunctionSpace(mesh, "HDiv Trace", 1)

hybrid = False

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
    'Iteration Limit': 100
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

#l1 variables
bk = Function(T)
dk = Function(T)
    
sig = Constant(1.0e-1)
lda = Constant(1.0)

U = FunctionSpace(mesh, "CG", 1)
u = Function(U).assign(0.2)


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

n = FacetNormal(mesh)

x, t = SpatialCoordinate(mesh)

c = Constant(0.2)
t0 = Constant(0.02)
H = lambda x: 0.5*(tanh(-x/t0) + 1)

I0 = H(abs(x-0.5 - c*t) - 0.2)
#I0 = conditional(abs(x-0.5 - c*t) < 0.2, 1.0, 0.0)
#I0 = conditional(abs(x-0.5) < 0.2, 1.0, 0.0)
#u = Function(U).interpolate(c)
b = as_vector([u,1])

thetaS = theta('+')
dthetaS = dtheta('+')
bkS = bk('+')
dkS = dk('+')

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
eqn += sig*lda*dtheta*(theta+bk-dk)*ds(rule=Vrule) + \
                    sig*lda*dthetaS*(thetaS+bks-dks)*dS(rule=Vrule)

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

cu = Constant(0.001) #metamorphosis penalty parameter (inverse)
J = assemble(1./2*z*z*dx + cu*1./2*v0*u*dx +
             sig*lda*0.5*(thetaS*thetaS*dS(rule=Vrule)
                      + theta*theta*ds(rule=Vrule)))
m = Control(v0)
b_ctrl = Control(bk)
d_ctrl = Control(dk)
Jhat = ReducedFunctional(J, m)
Jhat.optimize_tape()

# Define optimisation problem
problem = MinimizationProblem(Jhat)

solver = ROLSolver(problem, params_dict, inner_product="L2")

if hybrid:
    I, z, T = w.split()
else:
    z, I, T = w.split()

nits = 5
Ninner = 2

bkval = Function(T)
dkval = Function(T)

for it in range(nits):
    for inner in range(Ninner):
        z_opt = solver.solve()
        file0.write(I,z,u)
        dkval.assign( sign(theta+b)*max(abs(theta+b)-1/lda, 0) )
        d_ctrl.tape_value().assign(dkval)
    bkval.assign( bkval + theta - dkval )
    b_ctrl.tape_value().assign(bkval)
