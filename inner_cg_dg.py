from firedrake import *

nx = 40
Length = 1.0
Height = 1.0
ny = 40
mesh = PeriodicRectangleMesh(nx, ny, Length, Height,
                             direction="x",
                             quadrilateral=True)

VI = FunctionSpace(mesh, "DG", 1)
VZ = FunctionSpace(mesh, "CG", 3)

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

I,z = TrialFunctions(W)

aP = (dz*z + )*dx

zProb = NonlinearVariationalProblem(eqn, w, Jp=aP)
sparams = {'ksp_type':'gmres',
           'snes_linesearch':'basic', 
           'ksp_monitor':True,
           'ksp_converged_reason':True,
           'pc_type':'sor'}

parameters = {
    "ksp_type": "gmres",
    "ksp_monitor": True,
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "lu"
}

zSolver = NonlinearVariationalSolver(zProb,
                                     solver_parameters=parameters)

zSolver.solve()
