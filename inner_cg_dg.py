from firedrake import *

nx = 40
Length = 1.0
Height = 1.0
ny = 40
mesh = PeriodicRectangleMesh(nx, ny, Length, Height,
                             direction="x",
                             quadrilateral=True)

VI = FunctionSpace(mesh, "DG", 1)
VZ = FunctionSpace(mesh, "CG", 2)

zfirs = False
if zfirs:
    W = VZ * VI
    w = Function(W)
    z, I = split(w)
    dz, dI = TestFunctions(W)
else:
    W = VI * VZ
    w = Function(W)
    I, z = split(w)
    dI, dz = TestFunctions(W)

c = Constant(0.2)

U = FunctionSpace(mesh, "CG", 1)
u = Function(U).assign(c)

b = as_vector([u,1])

x, t = SpatialCoordinate(mesh)

I0 = conditional(abs(x-c*t) < 0.3, 1.0, 0.0)

n = FacetNormal(mesh)

eqn = (
    dz*z*dx + div(b*dz)*I*dx
    - inner(b*dz,n)*I0*ds
    + dI*div(b*z)*dx
)

zProb = NonlinearVariationalProblem(eqn, w)
parameters = {
    "ksp_type": "preonly",
    "mat_type": "aij",
    "pc_type": "lu"
}

zSolver = NonlinearVariationalSolver(zProb,
                                     solver_parameters=parameters)

zSolver.solve()
