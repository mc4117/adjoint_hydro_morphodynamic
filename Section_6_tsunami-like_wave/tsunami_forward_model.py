"""
Tsunami Test case
=======================

This file runs the forward model for the tsunami test case. The set-up used is that from the laboratory test case in 

[1] Kobayashi,  N.,  Lawrence,  A.R.,  2004.   Cross-shore  sediment  transport  under  breaking  solitarywaves.  Journal of Geophysical Research:  Oceans 109.  doi:10.1029/2003jc002084.
"""

import thetis as th
import numpy as np
import pylab as plt
import pandas as pd
import time
import datetime
import os

import matplotlib
font = {'size'   : 14}
matplotlib.rc('font', **font)


def tsunami_propagation(mesh2d, bathymetry_2d, uv, elev, sediment, dt, t_end, morfac, max_angle, init_flag = True, bedload_flag = False):
    
    # define tsunami wave
    H = 0.216
    h = 0.8
    C = 3.16
    eta_down = -0.0025
    tmax = 3.9

    def update_forcings(t_new):
        # update boundary condition
        elev_const.assign(tsunami_elev(t_new))

    x,y = th.SpatialCoordinate(mesh2d)

    # define function spaces
    V = th.FunctionSpace(mesh2d, 'CG', 1)

    # initialise velocity and elevation
    elev_init = th.Function(V).interpolate(th.Constant(-0.0025))
    uv_init = th.as_vector((1e-10, 0.0))

    # export interval in seconds
    t_export = 1

    # choose directory to output results
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    outputdir = 'outputs'+ st

    th.print_output('Exporting to '+outputdir)

    # set up solver
    solver_obj = th.solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.output_directory = outputdir

    options.check_volume_conservation_2d = True

    options.fields_to_export = ['uv_2d', 'elev_2d', 'sediment_2d', 'bathymetry_2d']

    options.timestepper_type = 'CrankNicolson'
    options.timestepper_options.implicitness_theta = 1.0
    options.norm_smoother = th.Constant(1/12*0.2)
    options.use_wetting_and_drying = True
    options.wetting_and_drying_alpha = th.Constant(2/12*0.2)
    options.quadratic_drag_coefficient = th.Constant(9.81/(65**2)) #chezy
    options.horizontal_viscosity = th.Constant(0.8)

    # specify which components of sediment model are being used
    options.sediment_model_options.solve_suspended_sediment = True
    options.sediment_model_options.use_bedload = bedload_flag
    options.sediment_model_options.solve_exner = True
    options.sediment_model_options.use_sediment_conservative_form = True
    
    # set sediment model parameters
    options.sediment_model_options.average_sediment_size = th.Constant(1.8e-4)
    options.sediment_model_options.bed_reference_height = th.Constant(0.00054)
    options.sediment_model_options.use_sediment_slide = True
    options.sediment_model_options.sed_slide_length_scale = th.Constant(0.2)
    options.horizontal_diffusivity = th.Constant(1)
    options.sediment_model_options.morphological_viscosity = th.Constant(1e-6)
    options.sediment_model_options.max_angle = th.Constant(max_angle)
    options.sediment_model_options.morphological_acceleration_factor = th.Constant(morfac)


    options.timestep = dt

    # tsunami elev function
    tsunami_elev = lambda t: H*(1/np.cosh(np.sqrt((3*H)/(4*h))*(C/h)*(t-tmax)))**2 + eta_down

    # set boundary conditions
    swe_bnd = {}

    elev_const = th.Constant(0.0)
    elev_const.assign(tsunami_elev(0.0))
    # apply tsunami-like wave as an input boundary condition
    swe_bnd[1] = {'elev': elev_const}

    solver_obj.bnd_functions['shallow_water'] = swe_bnd


    if init_flag:
        solver_obj.assign_initial_conditions(uv=uv_init, elev=elev_init)
    else:
        solver_obj.assign_initial_conditions(uv=uv, elev=elev, sediment = sediment)

    # run model
    solver_obj.iterate(update_forcings = update_forcings)

    return solver_obj.fields.bathymetry_2d, solver_obj.fields.uv_2d, solver_obj.fields.elev_2d, solver_obj.fields.sediment_2d, options

# define mesh
lx = 30
ly = 4
nx = lx*5
ny = np.int(ly*5)
mesh2d = th.RectangleMesh(nx, ny, lx, ly)

# define function spaces
V = th.FunctionSpace(mesh2d, 'CG', 1)

# define underlying bathymetry
init_bathymetry_2d = th.Function(V, name='Bathymetry')
x,y = th.SpatialCoordinate(mesh2d)

beach_profile = -x/12 + 131/120

init_bathymetry_2d.interpolate(th.conditional(x<3.5, th.Constant(0.8), beach_profile))
init = th.Function(V).interpolate(th.conditional(x<3.5, th.Constant(0.8), beach_profile))

# record initial bathymetry
xaxis = np.linspace(0, 29, 200)
init_bathlist = []
for i in xaxis:
    init_bathlist.append(-init.at([i, 1.2]))

bathymetry_2d = init_bathymetry_2d.copy(deepcopy=True)

init = True

for i in range(2):
    # simulate two waves with morphological scale factor of 4 so equivalent to 8 waves
    
    # run forward model
    bathymetry_out, uv, elev, sediment, options = tsunami_propagation(mesh2d, bathymetry_2d, None, None, None, dt = 0.025, t_end = 20, morfac = 4, max_angle = 22, init_flag = init, bedload_flag = False)
    bathymetry_2d = bathymetry_out.copy(deepcopy = True)
    
    # record bathymetry and plot
    bathlist = []
    for j in xaxis:
        bathlist.append(-bathymetry_2d.at([j, 1.2]))
    directory = os.path.dirname(os.path.abspath(__file__)) + "/"
    test4_p = pd.read_csv(directory + 'data_paper.csv', header = None)
    plt.plot(xaxis-3.5, init_bathlist)
    plt.plot(xaxis-3.5, bathlist)
    plt.scatter(test4_p[0], test4_p[1])
    plt.ylim([-0.5, 0.5])
    plt.show()
    
    init=False

# plot final bathymetry compared to initial bathymetry and experimental data
plt.plot(xaxis-3.5, init_bathlist, ':', label = 'Initial bedlevel')
plt.plot(xaxis-3.5, bathlist, label = 'Final bedlevel')
plt.scatter(test4_p[0], test4_p[1], label = 'Experimental data')
plt.ylim([-0.8, 0.8])
plt.legend()
plt.xlabel('x (m)')
plt.ylabel('height (m)')
plt.xlim([0, 20])
plt.show()
