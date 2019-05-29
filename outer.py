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

hybrid = True

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

U = FunctionSpace(mesh, "CG", 1)
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
    - div(b)*dz*I*dx
    - jump(b*dz,n)*thetaS*dS + inner(b*dz,n)*I0*ds
    +  dI*div(b*z)*dx
    - div(b)*z*dI*dx
    + jump(b*z,n)*dthetaS*dS
)
eqn += sig*dtheta*theta*ds

eqn1 = eqn + sig*eps*thetaS*dthetaS/(eps**2 + thetaS**2)**0.5*dS0
eqn0 = eqn + sig*thetaS*dthetaS*dS

sparams = {'ksp_type':'gmres',
           'ksp_converged_reason':True,
           'mat_type':'matfree',
           'pc_type':'python',
           'pc_python_type':'scpc.HybridSCPC',
           'hybrid_sc_ksp_type':'gmres',
           'hybrid_sc_ksp_converged_reason':False,
           'hybrid_sc_pc_type':'lu',
           'hybrid_sc_pc_factor_mat_solver_type':'mumps'}

sparams0 = {'ksp_type':'preonly',
            'snes_monitor':True,
            'snes_linesearch':'basic',
            'mat_type':'aij',
            'pc_type':'lu',
            'pc_factor_mat_solver_type':'mumps'}

z0Prob = NonlinearVariationalProblem(eqn0, w)
z0Solver = NonlinearVariationalSolver(z0Prob,
                                      solver_parameters=sparams)

zProb = NonlinearVariationalProblem(eqn1, w)
zSolver = NonlinearVariationalSolver(zProb,
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

cu = Constant(0.00001)

J = assemble(1./2*z*z*dx + cu*1./2*v0*v0*dx)
#J = assemble(1./2*z*z*dx)
m = Control(v0)
#m = Control(u)
Jhat = ReducedFunctional(J, m)

g_opt = minimize(Jhat, options={"disp": True, 'gtol':1.0e-4})

v0.assign(g_opt)
v0Solver.solve()
v1Solver.solve()
v2Solver.solve()
z0Solver.solve()
zSolver.solve()

if hybrid:
    I, z, T = w.split()
else:
    z, I, T = w.split()
f = File('IZ2.pvd')

z0Solver.solve()
f.write(I,z,u)
zSolver.solve()
f.write(I,z,u)
