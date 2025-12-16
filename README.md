# NYC-Taxi-Cab-Simulation-Project

This repository provides the code used in the testing for my CS5834 final report on simulating the NYC yellow taxi fleet to analyze electrification feasibility.


## Key files

Fleet simulation was run with `simulation_v5.py` and `gt_simulation_v4.py` for 2023 and 2013 simulation repesctively. The only difference between v4 and v5 of simulations are the inclusion of exchange station modeling.

Analysis of fleet simulations and their distance distributions is completed with `Simulation_v1.ipynb` and `Simulation_v2.ipynb`. The primary difference between versions is the inclusion of animated heat maps in version 2.

The intensive data cleaning and preparation is done by running `data/data formatting notebook.ipynb` chronologically. The start up overview section below goes over data download and setup for this.

Shift generation is completed in `Simulation_v2.ipynb` with functions from the `utils/shift_generation_{year}.py` files, which are designed to generate a parquet of simulation shifts for their specific year.

### Other files

Initial simulation development can be followed in the `Simulation_v1.ipynb` notebook while previous versions can be found in `_previous_sim_versions`





## Start up overview

The data files used for simulation even after cleaning and preparation are way too large to upload to GitHub, so you must download them from the sources and run the data formatting code one your own.

The data sources are as follows:
- 2013: [https://databank.illinois.edu/datasets/IDB-9610843](https://databank.illinois.edu/datasets/IDB-9610843)
- 2023: [https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

Once downloaded, move the files to copy the following folder structure:

```
repo/
└── data/
    ├── 2013_dist_batches/
    ├── 2023 yellow taxi/
    |   ├── filtered trip parquets
    |   ├── taxi_zones
    |   └── trip parquets
    |       └── **insert 2023 data: yellow_tripdata_2023-{num}.parquet for num 01 to 12**
    └── distilled taxi data
    └── FOIL data
    |   ├── decompressed
    |   |   └── FOIL2013
    |   |       └── **insert 2013 data: trip_data_{month}.csv for month 1 to 12**
    |   └── filtered
    └── sim_info
```

Next, run the cells within `data formatting notebook.ipynb` in order and it will filter the data and generate the distribution datasets. Please restart your kernel for each section to clear up system memory.

**NOTE:** This is an extremely intensive process with current batching designed to go up to 60 GB of RAM. Please adjust batch sizes for different steps if your machine has less available memory

<br>

Once data generation is complete, you should see the following files under the `data/sim_info` folder:
- driver_count_arrays.pkl
- driver_delta_arrays.pkl
- ground_truth_trips_raw.parquet
- in_between_miles_arrays.pkl
- in_between_minutes_arrays.pkl
- shift_arrays.pkl
- shift_information.parquet
- shift_start_counts_arrays.pkl
- shift_start_location_arrays.pkl
- shift_trips_raw.parquet

These files should be the entire raw information for the simulation codes to utilize. Further preperation such as enforcing a minimum connections threshold is completed within the simulation Python files.

<br>

With this, the data setup for the simulation should be complete

<br>


## Running a simulation

After completing the process of generation simulation data in the above section, there are only a few more steps to run a simulation:

1. Generate shift information with cell inside `Simulation_v2.ipynb`
  - This makes use of the `utils/shift_generation_{year}.py` files
2. Add and set the configuration in `test_config.yaml`
  - This should be done in the same folder the shift information was generated into
  - Examples can be found in included test folders, altough they do not include the large shift information files
3. Run simulation file
  - Command in anaconda prompt: `python simulation_v5.py {test_folder}`
  - All configuration is pulled from the yaml file inside the folder
  - Multiple simulations can be run simultaneous, however, their constant event logging can quickly overload memory, especially if taxis constantly go to low range. 




































