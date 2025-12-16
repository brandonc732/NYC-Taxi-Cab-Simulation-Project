# NYC-Taxi-Cab-Simulation-Project

This repository provides the code used in the testing for my CS5834 final report on simulating the NYC yellow taxi fleet to analyze electrification feasibility.


## Key files

Fleet simulation was run with `simulation_v5.py` and `gt_simulation_v4.py` for 2023 and 2013 simulation repesctively. The only difference between v4 and v5 of simulations are the inclusion of exchange station modeling.

Analysis of fleet simulations and their distance distributions is completed with `Simulation_v1.ipynb` and `Simulation_v2.ipynb`. The primary difference between versions is the inclusion of animated heat maps in version 2.

The intensive data cleaning and preparation is done by running `data/data formatting notebook.ipynb` chronologically. The start up overview section below goes over data download and setup for this.

Shift generation is completed with the `utils/shift_generation_{year}.py` files, which are designed to generate a parquet of simulation shifts for their specific year.

### Other files

Simulation development can be followed in the Simulation notebooks while previous versions can be found in `_previous_sim_versions`





## Start up overview





## Running a simulation




































