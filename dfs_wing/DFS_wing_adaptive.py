from dolfin import *; from mshr import *; 
import PyOmega_h as omega_h;
import time;
import logging; 
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('DOLFIN').setLevel(logging.WARNING)
set_log_level(logging.WARNING)
from numpy import linalg as LA

# Read Gmsh file
comm_world = omega_h.world()
mesh_osh = omega_h.gmsh_read_file(omega_h.path('wing_naca_3D_18.msh'), comm_world)
mesh_osh.balance()
mesh_osh.set_parting(omega_h.GHOSTED, 1)
omega_h.add_implied_metric_tag(mesh_osh)
mesh_osh.set_parting(omega_h.ELEM_BASED, 0)

XMIN = -1.52
XMAX = 3.8
YMIN = 1.35
YMAX = -1.35
ZMIN = 0.0
ZMAX = 1.35

# Simulation variables
T = 0.2          # final time
#num_steps = 10  # number of time steps
#dt = T / num_steps # time step size
mu = 1             # kinematic viscosity
eps = 1e-5 	       # epsilon

maxiter = 3
i=0

#Convert to dolfin
mesh = Mesh()


while(i < maxiter):
	# Import mesh into dolfin
	omega_h.mesh_to_dolfin(mesh, mesh_osh);

	# Define function spaces
	V = VectorFunctionSpace(mesh, "CG", 1)
	Q = FunctionSpace(mesh, "CG", 1)
	W_V = VectorFunctionSpace(mesh, "DG", 0)
	W = FunctionSpace(mesh, "DG", 0)

	# Boundary conditions, Dirichlet on inlet and outlet
	inflow  = DirichletBC(V, as_vector((1.0,0,0)), "x[0] < -1.52 + 1e-5")
	outflow = DirichletBC(Q, 0, " x[0] > 3.8 - 1e-5")
	noslip = Expression("x[0] > XMIN + eps && x[0] < XMAX - eps ? 1. : 0.", XMIN=XMIN, XMAX=XMAX, eps=eps, element = Q.ufl_element())
	bcu = [inflow]
	bcp = [outflow]

	# Define trial and test functions
	u = TrialFunction(V)
	v = TestFunction(V)
	p = TrialFunction(Q)
	q = TestFunction(Q)

	# Define functions for solutions at previous and current time steps
	u_n = Function(V)
	u_  = Function(V)
	p_n = Function(Q)
	p_  = Function(Q)
	u_out = Function(V)

	# Residual
	res_m = Function(V)
	res_c = Function(Q)

	# Define expressions used in variational forms
	n   = FacetNormal(mesh)
	f   = Constant((0, 0))

	h = CellDiameter(mesh)
	hmin = mesh.hmin()

	k = Constant(0.1 * hmin)
	dt = 0.1 * hmin
	k   = Constant(dt)
	nu = Constant(0.0)
	c1 = Constant(0.1)
	c2 = Constant(0.0)
	c3 = Constant(0.0)

	#theta=0.5
	um = 0.5*(u_ + u_n)
	d = c1 * h ** (2.0/2.0)
	d1 = .2*h**(3./2.)
	beta = Constant(100.0)

	# Momentum equation 
	r_m = (inner(u_ - u_n, v) / k) * dx \
		+ nu * inner(grad(um), grad(v)) * dx  \
		+ inner(grad(p_) + grad(um)*um, v) * dx  \
		+ d1*(inner(grad(p_) + grad(um)*um, grad(v)*um) + inner(div(um), div(v)))*dx \
		+ (1/h) * noslip * inner(um, n) * inner(v, n) * ds \
		+ (1/h) * noslip * inner(um, n) * inner(v, n) * ds 
		#+ beta*(1/h) * inner(um, n) * inner(v, n) * ds(102)
		
	J_mom = derivative(r_m, u_, u)
	F_mom = r_m 
	a_mom = J_mom 
	L_mom = action(J_mom,u_) - F_mom

	# Pressure equation and correction
	r_c = 2 * k * inner(grad(p_ - p_n), grad(q)) * dx \
		+ (inner(div(um), q))* dx \
		+ d*(inner(grad(p_) + grad(um)*um, grad(q)))*dx 
		#+ hmin * h * p * q * dx
	J_con = derivative(r_c, p_, p)
	F_con = r_c
	a_con = J_con 
	L_con = action(J_con, p_) - F_con
	  
	# Time-stepping
	t = 0

	# Fixed point iteration
	maxit = 10
	 
	step = 0

	while t < T:	
		
		# Update current time
		t += dt
		print("Solving t=", t, ", step=", step)
		step += 1
		
		start = time.time()
		for ite in range(maxit):
			
			solve(a_mom == L_mom, u_, bcu)
			solve(a_con == L_con, p_, bcp)
			
		end = time.time()
		
		file_u = File("u_" + str(step) + "_" + str(i) + ".pvd")
		file_p = File("p_" + str(step) + "_" + str(i) + ".pvd")
		file_u << u_
		file_p << p_
		
		u_n.assign(u_)
		p_n.assign(p_)	
		
		#break;

	# Import u from dolfin to omega_h
	omega_h.function_from_dolfin(mesh_osh, u_._cpp_object, "u")
	 
	# Set up metric, adaptivity parameters
	mesh_osh.set_parting(omega_h.GHOSTED, 1);
	metric_input = omega_h.MetricInput()
	source = omega_h.MetricSource(omega_h.VARIATION, 2e-3, "u")
	metric_input.add_source(source)
	metric_input.should_limit_lengths = True
	metric_input.max_length = 1.0 / 2.0
	metric_input.should_limit_gradation = True
	omega_h.generate_target_metric_tag(mesh_osh, metric_input) 
	opts = omega_h.AdaptOpts(mesh_osh)
	opts.verbosity = omega_h.EXTRA_STATS
	opts.min_quality_allowed = 0.1			
	# Adapt mesh
	while(omega_h.approach_metric(mesh_osh, opts)):
		omega_h.adapt(mesh_osh, opts)
	
	omega_h.vtk_write_parallel('naca_adapted_' + str(i), mesh_osh)

	i+=1
