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
mesh_osh = omega_h.gmsh_read_file('wing_naca_3D_18.msh', comm_world)

XMIN = -1.52
XMAX = 3.8
YMIN = 1.35
YMAX = -1.35
ZMIN = 0.0
ZMAX = 1.35

# Simulation variables
T = 10           # final time
num_steps = 10  # number of time steps
dt = T / num_steps # time step size
mu = 1             # kinematic viscosity
eps = 1e-5 	       # epsilon

#Convert to dolfin
mesh = Mesh()
omega_h.mesh_to_dolfin(mesh, mesh_osh);

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)
W_V = VectorFunctionSpace(mesh, "DG", 0)
W = FunctionSpace(mesh, "DG", 0)

class Noslip(SubDomain):
    def inside(self, x, on_boundary):
       return on_boundary

# Sub domain for inflow
class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < XMIN + DOLFIN_EPS and on_boundary

# Sub domain for outflow 
class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > XMAX - DOLFIN_EPS and on_boundary

# Sub domain for sides (y,z)
class Sides(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < XMIN + DOLFIN_EPS and x[1] > YMAX - DOLFIN_EPS and x[2] < ZMIN + DOLFIN_EPS and x[2] > ZMAX - DOLFIN_EPS and on_boundary
        
# Initialize sub-domain instances
noslip_sub = Noslip()
inflow_sub = Inflow()
outflow_sub = Outflow()
sides_sub = Sides()

# Initialize mesh function for boundary domains
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
inflow_sub.mark(boundaries, 1)
outflow_sub.mark(boundaries, 2)
sides_sub.mark(boundaries, 3)

# Boundary conditions, Dirichlet on inlet and outlet
#noslip  = DirichletBC(V, as_vector((0.0,0,0)), "(x[0] > -1.52 + 1e-5) && (x[0] < 3.8 - 1e-5)")
#noslip  = DirichletBC(V, as_vector((0,0,0)), boundaries, 0)
#noslip_sides = DirichletBC(V, as_vector((0,0,0)), boundaries, 3)
inflow  = DirichletBC(V, as_vector((1.0,0,0)), "x[0] < -1.52 + 1e-5 and on_boundary")
outflow = DirichletBC(Q, 0, " x[0] > 3.8 - 1e-5 and on_boundary")
#c = DirichletBC(V, as_vector((0,0,0)), " x[0] < 3.8 - DOLFIN_EPS and x[0] > -1.52 + DOLFIN_EPS and on_boundary")

noslip = Expression("x[0] > XMIN + eps && x[0] < XMAX - eps ? 1. : 0.", XMIN=XMIN, XMAX=XMAX, eps=eps, element = Q.ufl_element())
#wing = Expression("x[0] > XMIN + eps && x[0] < XMAX - eps ? 1. : 0.", XMIN=XMIN, XMAX=XMAX, YMIN=YMIN, YMAX=YMAX, ZMIN=ZMIN, ZMAX=ZMAX, eps=eps, element = Q.ufl_element())
bcu = [inflow]
bcp = [outflow]

#ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

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
k   = Constant(dt)

h = CellDiameter(mesh)
hmin = mesh.hmin()

k = Constant(0.1 * hmin)
dt = 0.1 * hmin
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
	
		R_m = (inner(u_ - u_n, v) / k) * dx \
			  + nu * inner(grad(u_), grad(v)) * dx  \
			  + inner(grad(p_) + grad(u_)*u_, v) * dx 
		R_c = (inner(div(u_), q)) * dx
		Res_m = assemble(R_m)
		Res_c = assemble(R_c)
		res_m.vector().set_local(Res_m)
		res_c.vector().set_local(Res_c)
		#print(type(res_c.vector().get_local()))
		#print(dir(res_c.vector().get_local()))
		
		print("U NORM", norm(u_), Res_m.get_local().size)
		print("P NORM", norm(p_), Res_c.get_local().size)
		print("RESIDUAL MOMENTUM max: ", max(abs(Res_m.get_local())), "NORM: ", norm(res_m), "NORM NUMPY", LA.norm(res_m.vector().get_local()))
		print("RESIDUAL CONTINUITY max: ", max(abs(Res_c.get_local())), "NORM: ", norm(res_c), "NORM NUMPY:", LA.norm(res_c.vector().get_local()))
		
		u_n.assign(u_)
		p_n.assign(p_)
		
	end = time.time()
	
	# Compute drag, lift
	D = noslip*p_*n[0]*ds
	L = noslip*p_*n[1]*ds

	# Assemble functionals over sub domain
	drag = assemble(D)
	lift = assemble(L)

	print("Lift: %f" % lift)
	print("Drag: %f" % drag)	
	
	file_u = File("u_" + str(step) + ".pvd")
	file_p = File("p_" + str(step) + ".pvd")
	file_u << u_
	file_p << p_
	#print("U NORM", norm(u_), Res_m.get_local().size)
	#print("P NORM", norm(p_), Res_c.get_local().size)
	#print("RESIDUAL MOMENTUM max: ", max(abs(Res_m.get_local())), "NORM: ", norm(res_m))
	#print("RESIDUAL CONTINUITY max: ", max(abs(Res_c.get_local())), "NORM: ", norm(Res_c))
	
	#break;

#Compute residual 


#Set metric 

#Adapt omega_h mesh

#Convert mesh to dolfin
