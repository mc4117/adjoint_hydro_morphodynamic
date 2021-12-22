"""
Meander Test case - Tangent linear model
=======================

Uses the tangent linear model to calculate the derivative of an output functional with respect to an 
uncertain scalar parameter.

The forward model set-up for this test case can be found in 
[1] Clare et al. (2020). Hydro-morphodynamics 2D modelling using a discontinuous Galerkin discretisation.
    Computers & Geosciences, 104658. https://doi.org/10.1016/j.cageo.2020.104658
"""

from thetis import *
from firedrake_adjoint import *

import numpy as np
import pylab as plt

import time
import datetime

# derivative flag - if true compute the derivative
test_derivative = True
# taylor test flag - if true conduct the taylor test to check derivative calculated correctly
taylor_test_flag = False

# choose which parameter is uncertain
ks_flag = True
d50_flag = False

def forward(ks, rhos, average_size, mesh2d, V):
    """
    This function runs the forward model simulation of the meander test case
    """
    def update_forcings_bnd(t_new):

        gradient_flux = AdjFloat((-0.053 + 0.02)/6000)
        gradient_flux2 = AdjFloat((-0.02+0.053)/(18000-6000))
        gradient_elev = AdjFloat((0.07342-0.02478)/6000)
        gradient_elev2 = AdjFloat((-0.07342+0.02478)/(18000-6000))
        elev_init_const = AdjFloat(-0.062 + 0.05436)

        # update flux and elevation boundary condition
        if t_new != float(t_old):
            t_old.assign(float(t_new))
            if t_new*morfac <= 6000:
                elev_constant.assign(gradient_elev*t_new*morfac + elev_init_const)
                flux_constant.assign((gradient_flux*t_new*morfac) - 0.02)
            else:
                flux_constant.assign((gradient_flux2*(t_new*morfac-6000)) - 0.053)
                elev_constant.assign(gradient_elev2*(t_new*morfac-18000) + elev_init_const)

    t_old = Constant(0.0)

    # define mesh
    x,y = SpatialCoordinate(mesh2d)

    # define function spaces
    R_1d = FunctionSpace(mesh2d, 'R', 0)
    DG_2d = FunctionSpace(mesh2d, 'DG', 1)
    vector_dg = VectorFunctionSpace(mesh2d, 'DG', 1)

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

    initial_bathymetry_2d = Function(V).interpolate(bathymetry_2d)
    diff_bathy = Function(V).interpolate(Constant(0.0))

    if ks_flag:
        # if uncertain parameter ks then must include initial hydrodynamics spin-up in the adjoint framework
        # because the hydrodynamics depend on ks explicitly
        
        # define initial elevation
        elev_init = Function(DG_2d).interpolate(0.0544 - bathymetry_2d)
        #  define initial velocity
        uv_init = Function(vector_dg).interpolate(as_vector((0.001,0.001)))

        # choose directory to output results
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        outputdir = 'outputs' + st

        print_output('Exporting to '+outputdir)

        t_end = 200

        # export interval in seconds
        t_export = np.round(t_end/40, 0)


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
        options.nikuradse_bed_roughness = Function(V).interpolate(conditional(y<-5, Constant(0.003), ks))
        # setting viscosity
        options.horizontal_viscosity = Constant(5*10**(-2))

        # crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
        options.timestepper_type = 'CrankNicolson'
        options.timestepper_options.implicitness_theta = 1.0
        options.timestep = 1

        # set boundary conditions
        left_bnd_id = 1
        right_bnd_id = 2

        swe_bnd = {}
        gradient_flux = AdjFloat((-0.053 + 0.02)/6000)
        gradient_flux2 = AdjFloat((-0.02+0.053)/(18000-6000))
        gradient_elev = AdjFloat((0.07342-0.02478)/6000)
        gradient_elev2 = AdjFloat((-0.07342+0.02478)/(18000-6000))
        elev_init_const = AdjFloat(-0.062 + 0.05436)

        swe_bnd[3] = {'un': Constant(0.0)}
        swe_bnd[1] = {'flux': Constant(-0.02)}
        swe_bnd[2] = {'elev': Constant(elev_init_const), 'flux': Constant(0.02)}

        solver_obj.bnd_functions['shallow_water'] = swe_bnd

        solver_obj.assign_initial_conditions(uv=uv_init, elev=elev_init)

        # run model
        solver_obj.iterate()

        uv = solver_obj.fields.uv_2d.copy(deepcopy = True)
        elev = solver_obj.fields.elev_2d.copy(deepcopy = True)

    else:
        # if uncertain parameter is d50 can use the previous initial hydrodynamics spin-up 
        # (created by meander_hydro.py) because the hydrodynamics do not explicitly depend on d50 
        
        # initialise velocity and elevation
        chk = DumbCheckpoint("hydrodynamics_meander_fine/elevation", mode=FILE_READ)
        elev = Function(DG_2d, name="elevation")
        chk.load(elev)
        chk.close()

        chk = DumbCheckpoint('hydrodynamics_meander_fine/velocity', mode=FILE_READ)
        uv = Function(vector_dg, name="velocity")
        chk.load(uv)
        chk.close()

    # Now run full hydro-morphodynamic simulation

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    outputdir = 'outputs'+ st

    # set parameters
    morfac = 50
    dt = 2
    end_time = 18000
    viscosity_hydro = Constant(5*10**(-2))

    # set up solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options

    # specify which components of sediment model are being used
    options.sediment_model_options.solve_suspended_sediment = False
    options.sediment_model_options.use_bedload = True
    options.sediment_model_options.solve_exner = True
    options.sediment_model_options.use_angle_correction = True
    options.sediment_model_options.use_slope_mag_correction = True
    options.sediment_model_options.use_secondary_current = True
    options.sediment_model_options.use_advective_velocity_correction = False
    options.sediment_model_options.morphological_viscosity = Constant(1e-6)

    options.sediment_model_options.average_sediment_size = Function(V).interpolate(conditional(y<-5, Constant(1e-3), average_size))
    options.sediment_model_options.bed_reference_height = Function(V).interpolate(conditional(y<-5, Constant(0.003), ks))
    options.sediment_model_options.sediment_density = rhos
    options.sediment_model_options.morphological_acceleration_factor = Constant(morfac)

    options.simulation_end_time = end_time/morfac
    options.simulation_export_time = options.simulation_end_time/40

    options.output_directory = outputdir
    options.check_volume_conservation_2d = True

    options.fields_to_export = ['uv_2d', 'elev_2d', 'bathymetry_2d']

    # using nikuradse friction
    options.nikuradse_bed_roughness = Function(V).interpolate(Constant(0.003)) #conditional(y<-5, Constant(0.003), ks))

    # set horizontal diffusivity parameter
    options.horizontal_viscosity = Constant(viscosity_hydro)

    # crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
    options.timestepper_type = 'CrankNicolson'
    options.timestepper_options.implicitness_theta = 1.0

    if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
        options.timestep = dt

    left_bnd_id = 1
    right_bnd_id = 2

    # set boundary conditions
    gradient_flux = AdjFloat((-0.053 + 0.02)/6000)
    gradient_flux2 = AdjFloat((-0.02+0.053)/(18000-6000))
    gradient_elev = AdjFloat((0.07342-0.02478)/6000)
    gradient_elev2 = AdjFloat((-0.07342+0.02478)/(18000-6000))
    elev_init_const = AdjFloat(-0.062 + 0.05436)

    swe_bnd = {}
    swe_bnd[3] = {'un': Constant(0.0)}

    flux_constant = Constant(-0.02)
    elev_constant = Constant(elev_init_const)

    swe_bnd[left_bnd_id] = {'flux': flux_constant}
    swe_bnd[right_bnd_id] = {'elev': elev_constant}

    solver_obj.bnd_functions['shallow_water'] = swe_bnd   

    solver_obj.assign_initial_conditions(uv=uv, elev=elev)

    # run model
    solver_obj.iterate(update_forcings = update_forcings_bnd)

    # calculate bed evolution
    diff_bathy.interpolate(initial_bathymetry_2d - solver_obj.fields.bathymetry_2d)

    return diff_bathy, solver_obj.options

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

x, y = SpatialCoordinate(mesh2d)
# define function spaces
V = FunctionSpace(mesh2d, 'CG', 1)

# define parameters
ks = Function(V).interpolate(Constant(0.003))
rhos = Constant(2650)
average_size = Function(V).interpolate(conditional(y<-5, Constant(0.0), Constant(1e-3)))

# run forward model
diff_bath, options = forward(ks, rhos, average_size, mesh2d, V)

# set up output functional ignoring the long inflow and outflow channels for simplicity 
diff_bath_constr = Function(V).project(conditional(y<1, Constant(0), diff_bath))
J = diff_bath_constr

if test_derivative:
    # calculate derivative
    if ks_flag:
        # set tangent linear model value
        h = Function(V)
        h.vector()[:] = 1e-4
        ks.block_variable.tlm_value = h

        tape = get_working_tape()
        tape.evaluate_tlm()
        
        dJdm = J.block_variable.tlm_value
        print(dJdm)

        # save dJdm
        checkpoint_dir = "meander_der"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        chk = DumbCheckpoint(checkpoint_dir + "/tlm_ks_fine", mode=FILE_CREATE)
        chk.store(dJdm, name="der")
        chk.close()        
        
        # visualise dJdm
        f = File('tlm_ks_fine.pvd')
        f.write(dJdm)         

    elif d50_flag:
        # set tangent linear model value
        h = Function(V)
        h.vector()[:] = 1e-4
        average_size.block_variable.tlm_value = h

        tape = get_working_tape()
        tape.evaluate_tlm()
        
        dJdm = J.block_variable.tlm_value
        print(dJdm)
        
        # save dJdm
        checkpoint_dir = "meander_der"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        chk = DumbCheckpoint(checkpoint_dir + "/tlm_d50_fine", mode=FILE_CREATE)
        chk.store(dJdm, name="der")
        chk.close()        
        
        # visualise dJdm
        f = File('tlm_d50_fine.pvd')
        f.write(dJdm)              
        
if taylor_test_flag:
    # check taylor test passes
    if d50_flag:
        # convert spatial field to single value for taylor test
        J_new = assemble(J*dx)
        # tell pyadjoint what the functional and uncertain parameter are
        Jhat = ReducedFunctional(J_new, Control(average_size))

        # set tlm value
        h = Function(V)
        h.vector()[:] = 1e-4
        average_size.block_variable.tlm_value = h

        g = average_size.copy(deepcopy=True)
        average_size.block_variable.tlm_value = h
        tape = get_working_tape()
        tape.evaluate_tlm()
        conv_rate = taylor_test(Jhat, g, h, dJdm=J_new.block_variable.tlm_value)
    
        if conv_rate > 1.9:
            print('*** test passed ***')
        else:
            print('*** ERROR: test failed ***')   
            
    if ks_flag:
        # convert spatial field to single value for taylor test
        J_new = assemble(J*dx)
        # tell pyadjoint what the functional and uncertain parameter are
        Jhat = ReducedFunctional(J_new, Control(ks))

        # set tlm value
        h = Function(V)
        h.vector()[:] = 1e-4
        ks.block_variable.tlm_value = h

        g = ks.copy(deepcopy=True)
        ks.block_variable.tlm_value = h
        tape = get_working_tape()
        tape.evaluate_tlm()
        conv_rate = taylor_test(Jhat, g, h, dJdm=J_new.block_variable.tlm_value)
    
        if conv_rate > 1.9:
            print('*** test passed ***')
        else:
            print('*** ERROR: test failed ***')   
