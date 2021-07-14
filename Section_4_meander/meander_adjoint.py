"""
Meander Test case - Adjoint
=======================

Uses the adjoint method to calculate the derivative of an output functional with respect to an 
uncertain spatially-varying parameter.

The forward model set-up for this test case can be found in 
[1] Clare et al. (2020). Hydro-morphodynamics 2D modelling using a discontinuous Galerkin discretisation.
    Computers & Geosciences, 104658. https://doi.org/10.1016/j.cageo.2020.104658
"""

# Note only one of the flags below should be set to True for each code-run.

# derivative flag - if true compute the derivative (and test that it has been calculated correctly)
test_derivative = False
# taylor test flag - if true conduct the taylor test (another check to check derivative calculated correctly)
taylor_test_flag = False
# perturb flag - if true then perturb uncertain parameter in the direction of derivative and compute difference
#                between perturbed bathymetry and original bathymetry
perturb = True

# import required packages
if not perturb:
    from firedrake_adjoint import *
else:
    from firedrake import *
    from firedrake_adjoint import AdjFloat

from thetis import *

import numpy as np
import pylab as plt

import time
import datetime

# choose which parameter is uncertain
ks_flag = False
d50_flag = True

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

        uv_hydro, elev = solver_obj.fields.solution_2d.split()
    else:
        # if uncertain parameter is d50 can use the previous initial hydrodynamics spin-up 
        # (created by meander_hydro.py) because the hydrodynamics do not explicitly depend on d50 
        
        # initialise velocity and elevation
        chk = DumbCheckpoint("hydrodynamics_meander_fine/elevation", mode=FILE_READ)
        elev = Function(DG_2d, name="elevation")
        chk.load(elev)
        chk.close()

        chk = DumbCheckpoint('hydrodynamics_meander_fine/velocity', mode=FILE_READ)
        uv_hydro = Function(vector_dg, name="velocity")
        chk.load(uv_hydro)
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
    options.nikuradse_bed_roughness = Function(V).interpolate(conditional(y<-5, Constant(0.003), ks))

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

    solver_obj.assign_initial_conditions(uv=uv_hydro, elev=elev)

    # run model
    solver_obj.iterate(update_forcings = update_forcings_bnd)

    return solver_obj.fields.bathymetry_2d, uv_hydro

# define mesh
mesh2d = Mesh("meander_fine.msh")

def snap_mesh_bnd_to_circle_arc(m, circle_arc_list, degree=2):

    #Snap mesh boundary nodes to a circle arc.
    
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
rhos = Constant(2650)

ks = Function(V).interpolate(Constant(0.003))
if ks_flag:
    ks_diff = Function(V).interpolate(Constant(0.003+3e-6))

average_size = Function(V).interpolate(conditional(y<-5, Constant(0.0), Constant(1e-3)))
if d50_flag:
    average_size_diff = Function(V).interpolate(conditional(y<-5, Constant(0.0), Constant(1e-3+1e-6)))

# run forward model
bath, uv_old = forward(ks, rhos, average_size, mesh2d, V)

# set up output functional ignoring the long inflow and outflow channels for simplicity 
# and using a smoothing factor
bath_constr = Function(V).project(conditional(y<1, Constant(0), bath))
J = assemble(((inner(bath_constr,bath_constr)+Constant(1e-6))**0.5)*dx)

print(J)

if test_derivative:
    # compute derivative of J with respect to uncertain parameter
    # and also check derivative computed correctly by comparing (J_h - J_0) with dJ/dm * h. 
    # For small enough h these should be equal

    if ks_flag:
        # tell pyadjoint what the functional and uncertain parameter are
        rf = ReducedFunctional(J, Control(ks))

        # replay tape. If J_0 is not equal to J (within a reasonable tolerance) then 
        # pyadjoint has not taped the model correctly
        J_0 = rf(ks)
        print(J_0)

        # calculate derivative
        der = rf.derivative()
        der_list = []
        for i in range(len(der.dat.data[:])):
            der_list.append(der.dat.data[i] * 3e-6)
        # play tape with slightly peturbed ks
        J_h = rf(ks_diff)

        # check (J_h - J_0) = dJ/dm * h within reasonable degree of tolerance
        print(sum(der_list))
        print((J_h - J_0))
        
        # save dJ/dm
        checkpoint_dir = "meander_der"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        chk = DumbCheckpoint(checkpoint_dir + "/der_ks_fine", mode=FILE_CREATE)
        chk.store(der, name="der")
        chk.close()        
        
        # output dJ/dm for visualisation
        f = File('der_ks_fine.pvd')
        f.write(der)        


    elif d50_flag:
        # tell pyadjoint what the functional and uncertain parameter are
        rf = ReducedFunctional(J, Control(average_size))

        # replay tape. If J_0 is not equal to J (within a reasonable tolerance) then 
        # pyadjoint has not taped the model correctly
        J_0 = rf(average_size)
        print(J_0)
        
        # calculate derivative        
        der = rf.derivative()

        # play tape with slightly peturbed d50
        J_h = rf(average_size_diff)

        der_list = []
        for i in range(len(der.dat.data[:])):
            der_list.append(der.dat.data[i] * 1e-6)
            
        # check (J_h - J_0) = dJ/dm * h within reasonable degree of tolerance
        print(sum(der_list))
        print(J_0 - J_h)            

        # save dJ/dm
        checkpoint_dir = "meander_der"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        chk = DumbCheckpoint(checkpoint_dir + "/der_d50_fine", mode=FILE_CREATE)
        chk.store(der, name="der")
        chk.close()        
        
        # output dJ/dm for visualisation
        f = File('der_d50_fine.pvd')
        f.write(der)        

    stop

if taylor_test_flag:
    # check that the taylor test passes for each uncertain parameter
    if ks_flag:
        # tell pyadjoint what the functional and uncertain parameter are
        rf = ReducedFunctional(J, Control(ks))
        # set peturbation in taylor test
        h = Function(V).project(Constant(0.25))
        # corresponds to taylor test values of 0.003 + [2.5e-3, 1.25e-3, 6.25e-4, 3.125e-4]
        conv_rate = taylor_test(rf, ks, h)
    elif d50_flag:
        # tell pyadjoint what the functional and uncertain parameter are
        rf = ReducedFunctional(J, Control(average_size))
        # set peturbation in taylor test
        h = Function(V).project(Constant(2e-3))
        # corresponds to taylor test values of 1e-3 + [2e-5, 1e-5, 5e-6, 2.5e-6]
        conv_rate = taylor_test(rf, average_size, h)

    if conv_rate > 1.9:
        print('*** test passed ***')
    else:
        print('*** ERROR: test failed ***')
        
    stop


if perturb:
    # peturb the uncertain parameter by 1e-6 times the derivative
    if d50_flag:
        # load in dJ/dm from the checkpoint created using the flag test_derivative = True
        chk = DumbCheckpoint("meander_der/der_d50_fine", mode=FILE_READ)
        der = Function(V, name="der")
        chk.load(der)
        chk.close()

        # peturb d50 by derivative
        average_size_diff = Function(V).interpolate(average_size + (der*1e-6))
        new_bath, new_uv = forward(ks, rhos, average_size_diff, mesh2d, V)
        
        # calculate difference between peturbed bed and unpeturbed bed and visualise
        diff = Function(V).interpolate(-new_bath+bath)
        f = File('diff_d50_fine.pvd')
        f.write(diff)
        
    elif ks_flag:
        # load in dJ/dm from the checkpoint created using the flag test_derivative = True
        chk = DumbCheckpoint("meander_der/der_ks_fine", mode=FILE_READ)
        der = Function(V, name="der")
        chk.load(der)
        chk.close()

        # peturb ks by derivative
        ks_diff = Function(V).interpolate(ks + (der*3e-6))
        new_bath, uv_new = forward(ks_diff, rhos, average_size, mesh2d, V)
        
        # calculate difference between peturbed bed and unpeturbed bed and visualise
        diff = Function(V).interpolate(-new_bath+bath)
        f = File('diff_ks_fine.pvd')
        f.write(diff)