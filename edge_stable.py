from firedrake import *
from firedrake_adjoint import *

nx = 150
Length = 1.0
Height = 1.0
ny = 150

mesh = PeriodicRectangleMesh(nx, ny, Length, Height,
                             direction="x",
                             quadrilateral=True)

VI = FunctionSpace(mesh, "CG", 1)
Vu = FunctionSpace(mesh, "CG", 1)

c = Constant(0.2)
gamma = Constant(0.1)

n = FacetNormal(mesh)

x, t = SpatialCoordinate(mesh)

#translating jump example
I0 = conditional(abs(x-0.5 - c*t) < 0.2, 1.0, 0.0)
bcs = [DirichletBC(VI, I0, (1,2))]
#trousers example
I0 = conditional(abs(x-0.5) < 0.2, 1.0, 0.0)
I1 = conditional(abs(x-0.4) < 0.1, 1.0, 0.0) + \
     conditional(abs(x-0.75) < 0.1, 1.0, 0.0)
bcs = [DirichletBC(VI, I0, 1),
       DirichletBC(VI, I1, 2)]

u = Function(Vu, name="Velocity")
b = as_vector([u,1])
I = Function(VI, name="Image")
phi = TestFunction(VI)

#mesh size (general formula from Houston for unstructured meshes)
h = avg(CellVolume(mesh))/FacetArea(mesh)
hpow = 2

eqn_inner = (
    inner(b, grad(phi))*inner(b, grad(I))*dx #consistent term
    + gamma*h**hpow*inner(jump(grad(I)), jump(grad(phi)))*dS #edge stabilisation
)

sparams0 = {'ksp_type':'preonly',
            'ksp_monitor':None,
            'pc_type':'lu',
            'pc_factor_mat_solver_type':'mumps'}

innerProb = NonlinearVariationalProblem(eqn_inner, I, bcs=bcs)
innerSolver = NonlinearVariationalSolver(innerProb,
                                         solver_parameters=sparams0)

w = TestFunction(Vu)
v = Function(Vu, name = "v = L*u") # v = L*u

alpha = Constant(0.1)
bw = as_vector([w,0])
eqn_v2u = (
    w*u*dx + alpha**2*u.dx(0)*w.dx(0)*dx
    - w*v*dx
    )

v2uProb = NonlinearVariationalProblem(eqn_v2u, u)
v2uSolver = NonlinearVariationalSolver(v2uProb,
                                         solver_parameters=sparams0)

tol = 1.0e10

v2uSolver.solve()
innerSolver.solve()

J = assemble(0.5*(v*v*dx
    + c*inner(b, grad(I))*inner(b, grad(I))*dx
    + c*gamma*h**hpow*inner(jump(grad(I)), jump(grad(I)))*dS #edge stabilisation
    ))


control = Control(v)
rf = ReducedFunctional(J, control)

# Define optimisation problem
problem = MinimizationProblem(rf)

# Now configure the optimisation solver
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
        'Gradient Tolerance': 1e-5,
        'Step Tolerance': 1e-16,
        'Relative Step Tolerance': 1e-10,
        'Iteration Limit': 100
    }
}
solver = ROLSolver(problem, params_dict, inner_product="L2")
z_opt = solver.solve()

v.assign(0.)
v2uSolver.solve()
innerSolver.solve()

file9 = File('vI.pvd')
file9.write(I,u,v)


v.assign(z_opt)
v2uSolver.solve()
innerSolver.solve()

file9.write(I,u,v)
