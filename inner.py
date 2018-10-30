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

eps = 1.0e-2
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

eqn = (
    dz*z*dx + div(b*dz)*I*dx
    - jump(b*dz,n)*thetaS*dS - inner(b*dz,n)*I0*ds
    +  dI*div(b*z)*dx
    - jump(b*z,n)*dthetaS*dS +
    sig*thetaS*dthetaS/sqrt(eps**2 + thetaS**2)*dS 
)

zProb = NonlinearVariationalProblem(eqn, w)
sparams = {'ksp_type':'preonly',
           'ksp_monitor':True,
           'mat_type':'aij',
           'pc_type':'lu',
           'pc_factor_mat_solver_type':'mumps'}
zSolver = NonlinearVariationalSolver(zProb,
                                     solver_parameters=sparams)

zSolver.solve()
