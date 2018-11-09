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

W = VZ * VI * T

w = Function(W)
z, I, theta = split(w)
dz, dI, dtheta = TestFunctions(W)

eps = Constant(1.0e-8)
c = Constant(0.2)
sig = Constant(1.0e-4)

U = FunctionSpace(mesh, "CG", 1)

n = FacetNormal(mesh)

x, t = SpatialCoordinate(mesh)

#I0 = conditional(abs(x-0.5 - c*t) < 0.2, 1.0, 0.0)
I0 = conditional(abs(x-0.5) < 0.2, 1.0, 0.0)
u = Function(U)
#u.interpolate(c)
u.interpolate(cos(pi*t))
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

bcs = [DirichletBC(W.sub(2), 0, (1,2))]

sparams = {'ksp_type':'gmres',
           'ksp_monitor':True,
           'ksp_converged_reason':True,
           'mat_type':'matfree',
           'pc_type':'python',
           'pc_python_type':'scpc.HybridSCPC',
           'hybrid_sc_ksp_type':'gmres',
           'hybrid_sc_ksp_monitor':True,
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

z, I, T = w.split()
f = File('IZ.pvd')

z0Solver.solve()
f.write(I,z)
zSolver.solve()
f.write(I,z)
