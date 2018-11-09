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

W = VZ * VI * T

w = Function(W)
z, I, theta = split(w)
dz, dI, dtheta = TestFunctions(W)

eps = Constant(1.0e-4)
c = Constant(0.2)
sig = Constant(1.0e-2)

U = FunctionSpace(mesh, "CG", 1)
u = Function(U)

n = FacetNormal(mesh)

x, t = SpatialCoordinate(mesh)

I0 = conditional(abs(x-0.5 - c*t) < 0.2, 1.0, 0.0)
#I0 = conditional(abs(x-0.5) < 0.2, 1.0, 0.0)
#u = Function(U).interpolate(c)
b = as_vector([u,1])

thetaS = theta('+')
dthetaS = dtheta('+')

dS0 = dS(degree=6)
ds0 = ds(degree=6)

eqn = (
    dz*z*dx + div(b*dz)*I*dx
    - jump(b*dz,n)*thetaS*dS + inner(b*dz,n)*I0*ds
    +  dI*div(b*z)*dx
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
           'hybrid_sc_pc_type':'lu'}

sparams0 = {'ksp_type':'preonly',
            'snes_monitor':True,
            'snes_linesearch':'basic',
            'mat_type':'aij',
            'pc_type':'lu'}

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

alpha = Constant(1.0e-2)

v0eqn = (du*v1 + alpha**2*inner(grad(du),grad(v1)) - du*v0)*dx
v1eqn = (du*v2 + alpha**2*inner(grad(du),grad(v2)) - du*v1)*dx
v2eqn = (du*u + alpha**2*inner(grad(du),grad(u)) - du*v2)*dx

v0Prob = NonlinearVariationalProblem(v0eqn, v1)
v1Prob = NonlinearVariationalProblem(v1eqn, v2)
v2Prob = NonlinearVariationalProblem(v2eqn, u)

luparams = {'ksp_type':'gmres',
            'pc_type':'sor'}

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

cu = Constant(1.0e-2)

J = assemble(1./2*z*z*dx + cu*1./2*v0*v0*dx)
m = Control(u)
Jhat = ReducedFunctional(J, m)

g_opt = minimize(Jhat, options={"disp": True, 'gtol':1.0e-3})

u.assign(g_opt)
z0Solver.solve()
zSolver.solve()

z, I, T = w.split()
f = File('IZ2.pvd')

z0Solver.solve()
f.write(I,z)
zSolver.solve()
f.write(I,z)
