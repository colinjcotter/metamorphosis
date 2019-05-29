from firedrake import *

nx = 40
Length = 1.0
Height = 1.0
ny = 40
mesh = PeriodicRectangleMesh(nx, ny, Length, Height,
                             direction="x",
                             quadrilateral=True)

VI = FunctionSpace(mesh, "CG", 1)
Vu = FunctionSpace(mesh, "CG", 1)
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

#I0 = conditional(abs(x-0.5 - c*t) < 0.2, 1.0, 0.0)
I0 = conditional(abs(x-0.5) < 0.2, 1.0, 0.0)
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

eqn0 = eqn + sig*thetaS*dthetaS*dS
L1term = sig*eps*thetaS*dthetaS/(eps**2 + thetaS**2)**0.5*dS0
eqn1 = eqn + L1term

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

z0Prob = NonlinearVariationalProblem(eqn0, w, bcs=bcs)
z0Solver = NonlinearVariationalSolver(z0Prob,
                                      solver_parameters=sparams)

zProb = NonlinearVariationalProblem(eqn1, w, bcs=bcs)
zSolver = NonlinearVariationalSolver(zProb,
                                     solver_parameters=sparams)

tol = 1.0e10
J = 0.5*z*z*dx + sig*eps**2*sqrt(eps**2 + thetaS**2)*dS

thetaS*dthetaS/(eps**2 + thetaS**2)*dS0
         
while tol > 1.0e-5:
    z0Solver.solve()
    zSolver.solve()

    tol = assemble(J)
    print(tol,"Tolerance")

z, I, T = w.split()
f = File('IZ.pvd')

z0Solver.solve()
f.write(I,z)
zSolver.solve()
f.write(I,z)
