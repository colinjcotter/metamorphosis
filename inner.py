from firedrake import *

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

W = VI * VZ * T

w = Function(W)
I, z, theta = split(w)
dI, dz, dtheta = TestFunctions(W)

eps = Constant(1.0)
c = Constant(0.2)
sig = Constant(1.0)

U = FunctionSpace(mesh, "CG", 1)
u = Function(U).assign(c)

b = as_vector([u,1])

n = FacetNormal(mesh)

x, t = SpatialCoordinate(mesh)

I0 = conditional(abs(x-c*t) < 0.3, 1.0, 0.0)

thetaS = theta('+')
dthetaS = dtheta('+')

dS0 = dS(degree=6)
ds0 = ds(degree=6)

eqn = (
    dz*z*dx + div(b*dz)*I*dx
    - jump(b*dz,n)*thetaS*dS - inner(b*dz,n)*I0*ds
    -  dI*div(b*z)*dx
    + jump(b*z,n)*dthetaS*dS
    + sig*thetaS*dthetaS*dS0
)

bcs = [DirichletBC(W.sub(2), 0, (1,2))]

zProb = NonlinearVariationalProblem(eqn, w, bcs=bcs)
sparams = {'ksp_type':'gmres',
           'snes_linesearch':'basic', 
           'ksp_monitor':True,
           'ksp_converged_reason':True,
           'mat_type':'matfree',
           'pc_type':'python',
           'pc_python_type':'scpc.HybridSCPC',
           'hybrid_sc_ksp_type':'preonly',
           'hybrid_sc_ksp_monitor':True,
           'hybrid_sc_pc_type':'lu'}
zSolver = NonlinearVariationalSolver(zProb,
                                     solver_parameters=sparams)

zSolver.solve()
