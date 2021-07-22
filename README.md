Adjoint hydro-morphodynamic model framework
================

This repository contains the model described in the paper

*Mariana C. A. Clare, Stephan C. Kramer, Colin J. Cotter and Matthew D. Piggott*, **Calibration, inversion and sensitivity analysis for hydro-morphodynamic models through the application of adjoint methods**, submitted to Computers & Geosciences.

Software requirements
-------------------------

1. Firedrake (www.firedrakeproject.org)
    * The version of the Firedrake model we use has been stored and can be downloaded from the following site: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5105703.svg)](https://doi.org/10.5281/zenodo.5105703).
2. Thetis (https://thetisproject.org/download.html)
    * The version of the Thetis model we use has been stored and can be downloaded from the following site: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5105623.svg)](https://doi.org/10.5281/zenodo.5105623) 
3. Pyadjoint (https://github.com/dolfin-adjoint/pyadjoint)
    * The version of the pyadjoint library we use has been stored and can be downloaded from the following site: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5105785.svg)](https://doi.org/10.5281/zenodo.5105785)
4. Python 3.6.x-3.8.x


Simulation scripts
------------------

* Section 4: Meander Test Case
    
    Reproduce tangent linear model results (Section 4.1) with:
```
#!bash
    $ python Section_4_meander/meander_tlm.py
```

    with test_derivative = True in that file. Note the visualisations created 
    by this script are Figure 3 in the paper.

   Reproduce adjoint methods results (Section 4.2) with:
```
#!bash
    $ python Section_4_meander/meander_adjoint.py
```

    If test_derivative = True, the visualisations created by this script are Figure 5 in the paper.
    If peturb = True, the visualisations created by this script are Figure 6 in the paper.

* Section 5: Migrating Trench Test Case

   Reproduce dual twin experiment results (Section 5.1) with :
```
#!bash
    $ python trench_dual_twin.py
``` 
    with Section_5_trench/minimize_flag = True in that file.
    
   Reproduce calibration results (Section 5.2) with :
```
#!bash
    $ python Section_5_trench/trench_calibration.py
``` 
    with minimize_flag = True in that file.
    
* Section 6: Tsunami-like Wave Test Case
    
    Reproduce forward model results (Section 6.1) with:
```
#!bash
    $ python Section_6_tsunami-like_wave/tsunami_forward_model.py
```

    Note the final plot created by this script is Figure 11 in the paper.

   Reproduce dual twin experiment results (Section 6.2) with :
```
#!bash
    $ python Section_6_tsunami-like_wave/tsunami_dual_twin.py
``` 
    with minimize_flag = True in that file.
    
   Reproduce inversion results (Section 6.3) with :
```
#!bash
    $ python Section_6_tsunami-like_wave/tsunami_optimum.py
``` 
    with minimize_flag = True in that file.
