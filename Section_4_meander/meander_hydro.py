"""
Meander Test case
=======================
Solves the initial hydrodynamics simulation of flow around a 180 degree bend replicating
lab experiment 4 in Yen & Lee (1995).

Note this is not the main run-file and is just used to create an initial checkpoint for
the morphodynamic simulation.

For more details of the test case set-up see
[1] Clare et al. (2020). Hydro-morphodynamics 2D modelling using a discontinuous Galerkin discretisation.
    Computers & Geosciences, 104658. https://doi.org/10.1016/j.cageo.2020.104658
"""

from thetis import *

import numpy as np
import time

# define mesh
mesh2d = Mesh("meander_fine.msh")

def snap_mesh_bnd_to_circle_arc(m, circle_arc_list, degree=2):
    """
    Snap mesh boundary nodes to a circle arc.
    """
    # make new high degree coordinate function
    V = VectorFunctionSpace(m, 'CG', degree)
    new_coords = Function(V)
    x, y = SpatialCoordinate(m)
    new_coords.interpolate(as_vector((x, y)))
    for bnd_id, x0, y0, radius in circle_arc_list:
        # calculate new coordinates on circle arc
        xy_mag = sqrt((x - x0)**2 + (y - y0)**2)
        new_x = (x - x0)/xy_mag*radius + x0
        new_y = (y - y0)/xy_mag*radius + y0
        circle_coords = as_vector((new_x, new_y))
        bc = DirichletBC(V, circle_coords, bnd_id)
        bc.apply(new_coords)
    # make a new mesh
    new_mesh = mesh.Mesh(new_coords)
    return new_mesh


# define circle boundary arcs: bnd_id, x0, y0, radius
circle_arcs = [
    (4, 4.5, 2.5, 4.5),
    (5, 4.5, 2.5, 3.5),
]
mesh2d = snap_mesh_bnd_to_circle_arc(mesh2d, circle_arcs)

x,y = SpatialCoordinate(mesh2d)

# define function spaces
V = FunctionSpace(mesh2d, 'CG', 1)
P1_2d = FunctionSpace(mesh2d, 'DG', 1)
vectorP1_2d = VectorFunctionSpace(mesh2d, 'DG', 1)

# define underlying bathymetry
bathymetry_2d = Function(V, name='Bathymetry')
gradient = Constant(0.0035)
L_function = Function(V).interpolate(conditional(x > 5, pi*4*((pi/2)-acos((x-5)/(sqrt((x-5)**2+(y-2.5)**2))))/pi,
                                                 pi*4*((pi/2)-acos((-x+5)/(sqrt((x-5)**2+(y-2.5)**2))))/pi))
bathymetry_curve = Function(V).interpolate(conditional(y > 2.5,
                                                       conditional(x < 5, (L_function*gradient),
                                                       -(L_function*gradient)), 0))
init = max(bathymetry_curve.dat.data[:])
final = min(bathymetry_curve.dat.data[:])
bathymetry_straight = Function(V).interpolate(conditional(x <= 5,
                                              conditional(y <= 2.5, gradient*abs(y - 2.5) + init, 0),
                                              conditional(y <= 2.5, - gradient*abs(y - 2.5) + final, 0)))
bathymetry_2d = Function(V).interpolate(-bathymetry_curve - bathymetry_straight)

# define initial elevation
elev_init = Function(P1_2d).interpolate(0.0544 - bathymetry_2d)

# define initial velocity
uv_init = Function(vectorP1_2d).interpolate(as_vector((0.001,0.001)))

# choose directory to output results
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

print_output('Exporting to '+outputdir)

t_end = 200

# export interval in seconds
t_export = np.round(t_end/40, 0)

# define parameters
average_size = 10**(-3)
ksp = Constant(3*average_size)

# set up solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir

options.check_volume_conservation_2d = True

options.fields_to_export = ['uv_2d', 'elev_2d']
options.solve_tracer = False
options.use_lax_friedrichs_tracer = False

# using nikuradse friction
options.nikuradse_bed_roughness = ksp

# setting viscosity
options.horizontal_viscosity = Constant(5*10**(-2))

# crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
options.timestepper_type = 'CrankNicolson'
options.timestepper_options.implicitness_theta = 1.0

if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestep = 1

# set boundary conditions
left_bnd_id = 1
right_bnd_id = 2

swe_bnd = {}
elev_init_const = (-0.062 + 0.05436)

swe_bnd[3] = {'un': Constant(0.0)}
swe_bnd[1] = {'flux': Constant(-0.02)}
swe_bnd[2] = {'elev': Constant(elev_init_const), 'flux': Constant(0.02)}

solver_obj.bnd_functions['shallow_water'] = swe_bnd

solver_obj.assign_initial_conditions(uv=uv_init, elev=elev_init)

# run model
solver_obj.iterate()

uv, elev = solver_obj.fields.solution_2d.split()

# store hydrodynamics for next simulation
checkpoint_dir = "hydrodynamics_meander_fine"

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
chk = DumbCheckpoint(checkpoint_dir + "/velocity", mode=FILE_CREATE)
chk.store(uv, name="velocity")
chk.close()
chk = DumbCheckpoint(checkpoint_dir + "/elevation", mode=FILE_CREATE)
chk.store(elev, name="elevation")
chk.close()

