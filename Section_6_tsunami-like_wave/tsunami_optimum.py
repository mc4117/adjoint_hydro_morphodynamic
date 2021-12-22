"""
Tsunami Test case - Optimum parameter estimation
=======================

Uses the adjoint method to invert for uncertain parameters in the tsunami test case.

The true values in this file come from the experimental data.

The set-up used is that from the laboratory test case in 

[1] Kobayashi,  N.,  Lawrence,  A.R.,  2004.   Cross-shore  sediment  transport  under  breaking  solitarywaves.  Journal of Geophysical Research:  Oceans 109.  doi:10.1029/2003jc002084.
"""

from firedrake_adjoint import *
from thetis import *
from thetis.configuration import *
import numpy as np
import pandas as pd
import os
import time
import datetime

# Note only one of the flags below should be set to True for each code-run.
# The first two flags check that the adjoint model is working properly and the third flag runs the calibration

# derivative flag - if true compute the derivative (and test that it has been calculated correctly)
test_derivative = False
# taylor test flag - if true conduct the taylor test (another check to check derivative calculated correctly)
taylor_test_flag = False
# minimization flag - run optimisation algorithm to calibrate for uncertain parameters
minimize_flag = True

# write outputs to text file
def eval_callback(value):
    f= open("data_abs5e-6.txt","a")
    f.write(str([i.dat.data[:] for i in value]))
    f.write('\n')
    f.close()

def eval_post(functional_value, value):
    f= open("data_abs5e-6.txt","a")
    f.write(str(functional_value))
    f.write('\n')
    f.close()
    print(functional_value)


# define mesh
lx = 30
ly = 4
nx = lx*5
ny = np.int(ly*5)
mesh2d = RectangleMesh(nx, ny, lx, ly)

# define function spaces
V = get_functionspace(mesh2d, 'CG', 1)
P1_2d = get_functionspace(mesh2d, 'DG', 1)

# define underlying bathymetry
bathymetry_2d = Function(V, name='Bathymetry')
x,y = SpatialCoordinate(mesh2d)

beach_profile = -x/12 + 131/120

bathymetry_2d.interpolate(conditional(x<3.5, Constant(0.8), beach_profile))
init = Function(V).interpolate(conditional(x<3.5, Constant(0.8), beach_profile))

# define tsunami wave
H = AdjFloat(0.216)
h = AdjFloat(0.8)
C = AdjFloat(3.16)
eta_down = AdjFloat(-0.0025)
tmax = AdjFloat(3.9)

def tsunami_elev(H, t):
    return H*(cosh(sqrt((3*H)/(4*h))*(C/h)*(t-tmax)))**(-2) + eta_down

def forward(bathymetry, elev_list, uv_2d, elev_2d, tval):
    """
    This function runs the forward model
    """
    dt = 0.025
    morfac = 4
    max_angle = 22

    bath_list  = []
    def update_forcings(t_new):
        # update boundary condition
        if t_old.dat.data[:] != t_new:
            new_counter[0] += 1
            elev_const.assign(elev_list[new_counter[0]])
            t_old.assign(t_new)

    x,y = SpatialCoordinate(mesh2d)

    # define function spaces
    V = get_functionspace(mesh2d, 'CG', 1)
    P1_2d = get_functionspace(mesh2d, 'DG', 1)

    new_counter = [0]
    t_old = Constant(0.0)

    t_end = tval

    # export interval in seconds
    t_export = 1

    # choose directory to output results
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    outputdir = 'outputs'+ st

    print_output('Exporting to '+outputdir)

    # set up solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry)
    options = solver_obj.options
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.output_directory = outputdir
    options.check_volume_conservation_2d = True
    options.fields_to_export = ['uv_2d', 'elev_2d', 'sediment_2d', 'bathymetry_2d']

    # crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
    options.timestepper_type = 'CrankNicolson'
    options.timestepper_options.implicitness_theta = 1.0
    options.norm_smoother = Constant(1/12*0.2)
    options.use_wetting_and_drying = True
    options.wetting_and_drying_alpha = Constant(2/12*0.2)
    options.quadratic_drag_coefficient = Constant(9.81/(65**2)) #chezy
    options.horizontal_viscosity = Constant(0.8)

    # specify which components of sediment model are being used
    options.sediment_model_options.solve_suspended_sediment = True
    options.sediment_model_options.use_bedload = False  # according to Tehranirad
    options.sediment_model_options.solve_exner = True
    options.sediment_model_options.use_sediment_conservative_form = True
    
    # set key sediment parameters
    options.sediment_model_options.average_sediment_size = Constant(1.8e-4)
    options.sediment_model_options.bed_reference_height = Constant(0.00054) #2)
    options.sediment_model_options.use_sediment_slide = True
    options.sediment_model_options.meshgrid_size = Constant(0.2)
    options.horizontal_diffusivity = Constant(1)
    options.sediment_model_options.morphological_viscosity = Constant(1e-6)
    options.sediment_model_options.max_angle = Constant(max_angle)
    options.sediment_model_options.morphological_acceleration_factor = Constant(morfac)

    options.timestep = dt

    # set boundary conditions
    swe_bnd = {}
    elev_const = Constant(0.0)
    elev_const.assign(elev_list[0])
    # apply tsunami-like wave as an input boundary condition
    swe_bnd[1] = {'elev': elev_const}

    solver_obj.bnd_functions['shallow_water'] = swe_bnd

    # set initial conditions
    solver_obj.assign_initial_conditions(uv=uv_2d, elev=elev_2d)

    # run model
    solver_obj.iterate(update_forcings = update_forcings)
    return solver_obj.fields.uv_2d, solver_obj.fields.elev_2d, solver_obj.fields.bathymetry_2d

# set initial wave
elev_guess = []

times = np.linspace(0, 20, 801)

for i in times:
    if i < 2.5:
        elev_guess.append(Constant(tsunami_elev(H, i)))
    elif i < 5.325:
        elev_guess.append(Constant(0.05))
    else:
        elev_guess.append(Constant(tsunami_elev(H, i)))

# initialise hydrodynamics
elev_init = Function(P1_2d).interpolate(Constant(-0.0025))
elev_init2 = Function(P1_2d).interpolate(Constant(-0.0025))
uv_init = as_vector((1e-10, 0.0))
uv_init2 = as_vector((1e-10, 0.0))

# Run forward model twice to simulate two waves. 
# As morphological scale factor is 4 this is equivalent to eight waves
uv, elev, bath = forward(init, elev_guess, uv_init, elev_init, tval = 20)
final_uv, final_elev, new_bath = forward(bath, elev_guess, uv_init2, elev_init2, tval = 20)

# read in experimental data
directory = os.path.dirname(os.path.abspath(__file__)) + "/"
data = pd.read_csv(directory + 'data_paper.csv', header = None)

# calculate error between model bathymetry and experimental data
form = 0
for i in range(len(data)):
    bump_func = Function(V).project((1/0.01)*exp(-0.5*((x-3.5-data.iloc[i][0])/0.01)**2))
    normaliser = inner(bump_func, bump_func)*dx
    form += 0.5*inner(data.iloc[i][1]*bump_func + new_bath*bump_func, data.iloc[i][1]*bump_func + new_bath*bump_func)/(assemble(normaliser))

J_new = assemble(form*dx)

# Tikhonov regularisation terms

# abs norm
identity = Function(V).interpolate(Constant(5e-6))
final_sum = [assemble(((i**2)**0.5)*identity*dx) for i in elev_guess[100:213]]

# continuity norm
identity2 = Function(V).interpolate(Constant(5e-2))
conn_sum = [assemble((elev_guess[i]-elev_guess[i-1])*(elev_guess[i]-elev_guess[i-1])*identity2*dx) for i in range(1, len(elev_guess))]

# assemble output functional
J_fin = J_new + sum(final_sum) + sum(conn_sum)
print(J_fin)

elev_diff = [Constant(i.dat.data[0]+0.001) for i in elev_guess]

if test_derivative:
    # compute derivative of J with respect to uncertain parameter
    # and also check derivative computed correctly by comparing (J_h - J_0) with dJ/dm * h. 
    # For small enough h these should be equal
    
    # tell pyadjoint what the functional and uncertain parameters are
    rf = ReducedFunctional(J_fin, [Control(p) for p in elev_guess[100:213]])

    # replay tape. If J_0 is not equal to J (within a reasonable tolerance) then 
    # pyadjoint has not taped the model correctly
    J_0 = rf(elev_guess[100:213])
    print(J_0)

    # calculate derivative
    der = rf.derivative()
    der_sum = sum([der[i].dat.data[:]*(elev_diff[100+i].dat.data[:]-elev_guess[100+i].dat.data[:]) for i in range(len(der))])

    # play tape with slightly peturbed parameters
    J_h = rf(elev_diff[100:213])

    # check (J_h - J_0) = dJ/dm * h within reasonable degree of tolerance
    print(der_sum)
    print(J_h - J_0)

    stop

if taylor_test_flag:
    # check the taylor test passes
    
    # tell pyadjoint what the functional and uncertain parameters are
    rf = ReducedFunctional(J_fin, [Control(p) for p in elev_guess[100:213]])
    # peturbations used in taylor test
    H_h = AdjFloat(0.005)
    new_h = [Constant(tsunami_elev(H_h, i)) for i in times]
    conv_rate = taylor_test(rf, elev_guess[100:213], new_h[100:213])

    if conv_rate > 1.9:
        print('*** test passed ***')
    else:
        print('*** ERROR: test failed ***')

if minimize_flag:
    # tell pyadjoint what the functional and uncertain parameters are    
    rf = ReducedFunctional(J_fin, [Control(p) for p in elev_guess[100:213]], eval_cb_pre = eval_callback, eval_cb_post = eval_post)
    # compute minimisation algorithm to invert for uncertain values
    min_value = minimize(rf, options={'gtol':  1e-200, 'ftol': 1e-10, 'maxfun': 1000}, bounds = [[Constant(-0.0025) for _ in times[100:213]], [Constant(0.25) for _ in times[100:213]]])
