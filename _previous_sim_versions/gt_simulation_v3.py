import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import time
import yaml

import os
import sys
from pathlib import Path

import numpy as np
from numba import njit
from numba.typed import List as NumbaList



"""
Function to convert 2D dataframe of numpy arrays to the following:
- large 1D array of values
- 2D array of bin starts within the 1D array
- 2D array of bin lengths within the 1D array
- If a cell entry is an empty array, then it will be replaced by one entry of np.inf (should only apply to inbetween variables)

"""

def ragged_to_padded(arrs, n, fill=-1, dtype=np.int32):
    out = np.full((len(arrs), n), fill, dtype=dtype)
    for i, row in enumerate(arrs):
        k = min(len(row), n)          # truncate if longer than n
        if k:
            out[i, :k] = row[:k]
    return out


def convert_array_df(array_df):

    #print("columns:", array_df.columns)
    #print("rows:", array_df.index)

    missing_cols = check_series_is_full(array_df.columns)
    missing_rows = check_series_is_full(array_df.index)

    if len(missing_rows) > 0 or len(missing_cols) > 0:
        print("Filling in missing columns and rows with empty arrays")
        print("missing cols:", missing_cols)
        print("missing rows:", missing_rows)

        array_df = fill_df_missing_cols_rows(array_df)

    num_rows, num_cols = array_df.shape

    starts  = np.zeros((num_rows, num_cols), dtype=np.int64)
    lengths = np.zeros((num_rows, num_cols), dtype=np.int64)

    flat = []
    
    curr = 0
    for i in range(num_rows):
        for j in range(num_cols):

            arr = array_df.iat[i, j]

            # if the array is empty, just replace it with np.inf array
            if arr is None or len(arr) == 0:
                arr = np.array([np.inf])

            arr = np.asarray(arr)

            starts[i, j] = curr
            lengths[i, j]   = arr.size

            flat.append(arr)
            curr += arr.size

    flat = np.concatenate(flat)

    return {'values' : flat, 'starts' : starts, 'lengths' : lengths, 
            'min_row' : array_df.index.min(), 'min_col' : array_df.columns.min(),
            'cols' : array_df.columns, 'rows' : array_df.index}


def check_series_is_full(series):
    missing = set(range(series.min(), series.max() + 1)) - set(series)
    
    return missing

def fill_df_missing_cols_rows(df):
    # create indexes for full range
    full_index  = range(df.index.min(),   df.index.max()   + 1)
    full_columns = range(df.columns.min(), df.columns.max() + 1)

    # re-index to this complete index
    df = df.reindex(index=full_index, columns=full_columns)
    
    # this created NaNs, which we can fill with empty arrays
    df = df.map(lambda v: np.array([])    if     (v is None or (isinstance(v, float)     and np.isnan(v)))         else v)

    return df


"""
Function to load information for sampling.

This includes the following variables:

- inbetween_miles_arrays:   (PU, DO)      #NEEDS MULTIPLICATION
- inbetween_minutes_arrays: (PU, DO)

- shift_durations: (hour,day)
- shift_start_locations: (hour, day)

- driver_count_arrays: (hour, day)
- driver_count_deltas: (hour, day)


To speed up sampling, each variable distribution will be given by:
- large 1D array of values
- 2D array of bin starts within the 1D array
- 2D array of bin lengths within the 1D array
- If a cell entry is an empty array, then it will be replaced by one entry of np.inf (should only apply to inbetween variables)

NOTE: Everything is converted into seconds or miles

"""
def load_sampling_stuff():
    import gc

    print("Loading sampling info:")
    
    print("Shift durations...")
    shift_durations = pd.read_pickle("data/sim_info/shift_arrays.pkl")
    shift_durations_info = convert_array_df(shift_durations)
    del shift_durations
    gc.collect()

    print("Shift start locations...")
    shift_start_locations = pd.read_pickle("data/sim_info/shift_start_location_arrays.pkl")
    shift_start_locations_info = convert_array_df(shift_start_locations)
    del shift_start_locations
    gc.collect()
    

    print("driver counts...")
    driver_counts = pd.read_pickle("data/sim_info/driver_count_arrays.pkl")
    driver_counts_info = convert_array_df(driver_counts)
    del driver_counts
    gc.collect()

    print("driver count deltas...")
    driver_count_deltas = pd.read_pickle("data/sim_info/driver_delta_arrays.pkl")
    driver_count_deltas_info = convert_array_df(driver_count_deltas)
    del driver_count_deltas
    gc.collect()
    

    print("In between miles...")
    in_between_miles = pd.read_pickle("data/sim_info/in_between_miles_arrays.pkl")
    in_between_miles_info = convert_array_df(in_between_miles)
    del in_between_miles
    gc.collect()

    print("In between durations...")
    in_between_durations = pd.read_pickle("data/sim_info/in_between_minutes_arrays.pkl")
    in_between_durations_info = convert_array_df(in_between_durations)
    del in_between_durations
    gc.collect()

    
    print("Converting measurements")

    # convert times to seconds
    shift_durations_info['values']      = shift_durations_info['values'] * 3600      # originally was hours
    in_between_durations_info['values'] = in_between_durations_info['values'] * 60   # originally was minutes

    # convert shift start locations to int16
    shift_start_locations_info['values'] = shift_start_locations_info['values'].astype(np.int16)

    # convert driver count stuff to int16
    driver_counts_info['values'] = driver_counts_info['values'].astype(np.int16)
    driver_count_deltas_info['values'] = driver_count_deltas_info['values'].astype(np.int16)

    # apply LS equation from oak ridge paper to in between miles
    in_between_miles_info['values'] = in_between_miles_info['values'] * 1.4413 + 0.1383


    return {
        'shift_durations'       : shift_durations_info,
        'shift_start_locations' : shift_start_locations_info,
        'in_between_miles'      : in_between_miles_info,
        'in_between_durations'  : in_between_durations_info,
        'driver_counts'         : driver_counts_info,
        'driver_count_deltas'   : driver_count_deltas_info
    }





def get_PU_to_DO_connections(threshold):
    in_between_df = pd.read_pickle("data/sim_info/in_between_minutes_arrays.pkl")

    #in_between_df.head()
    
    in_between_counts = in_between_df.map(len)
    del in_between_df
    
    #in_between_counts.head()

    # set any counts below the threshold to zero
    counts_thresholded = in_between_counts.where(in_between_counts >= threshold, 0)
    
    #counts_thresholded.head()
    
    
    # go through each pickup location and make a list of dropoff locations that can service them
    PULocationIDs = counts_thresholded.index
    
    PU_DO_location_connectors= [[] for _ in range(PULocationIDs.min(), PULocationIDs.max()+1)]
    
    
    for PU_location in PULocationIDs:
    
        # get the row of dropoff zones that can service the current pickup
        connections_row = counts_thresholded.loc[PU_location]
    
        # sort the row entries by number of connections
        connections_row = connections_row[connections_row > 0].sort_values(ascending=False)
        
        # get a list of columns above zero
        DO_connections = connections_row.index.tolist()
    
        if PU_location in DO_connections:
            DO_connections.remove(PU_location)
        else:
            #print(DO_location)
            pass
    
        PU_DO_location_connectors[PU_location - PULocationIDs.min()] = DO_connections
    
    return {'data' : PU_DO_location_connectors,
            'min_locationID' : PULocationIDs.min()}












# NUMBA optimization for parimary search
@njit(cache=True)
def primary_search_numba(
    pickup_idx,
    pickup_time_seconds,
    trip_distance,
 
    taxi_bank, 
    in_between_starts,
    in_between_lengths,
    in_between_miles,
    in_between_seconds,

    primary_rejects_log #(N, 9) numpy zero array
):
    C_TIME=0; C_ID=1; C_CUR_LOC=2; C_RANGE=3; C_END=4; C_TG=5; C_PU=6; C_TRIP=7; C_IB=8
    
    # sample ib_miles and ib_dist
    start = in_between_starts[pickup_idx][pickup_idx]
    length = in_between_lengths[pickup_idx][pickup_idx]

    sample_index = np.random.randint(0, length)

    ib_miles = in_between_miles[start + sample_index]
    ib_time =  in_between_seconds[start + sample_index]

    tg_distance = (trip_distance + ib_miles)

    # search through taxi's in location array until reach np.inf
    j = 1 # start after first marker
    curr_reject_idx = 0
    while taxi_bank[pickup_idx, j, TIME] < gravestone + 1:

        # haven't reached end indication

        if (taxi_bank[pickup_idx, j, TIME] <= pickup_time_seconds and
            taxi_bank[pickup_idx, j, END] > pickup_time_seconds):

                # the taxi has valid pickup and end shift times.
                # check if it has enough range and log if it rejects from not having enough
                if taxi_bank[pickup_idx, j, RANGE] >= tg_distance:
                    return pickup_idx, j, ib_miles, ib_time, curr_reject_idx

                else:
                    # add info to reject log
                    primary_rejects_log[curr_reject_idx, C_TIME]    = pickup_time_seconds             # requested pickup time
                    primary_rejects_log[curr_reject_idx, C_ID]      = taxi_bank[pickup_idx, j, ID]    # taxi id
                    primary_rejects_log[curr_reject_idx, C_CUR_LOC] = pickup_idx + 1                  # current location
                    primary_rejects_log[curr_reject_idx, C_RANGE]   = taxi_bank[pickup_idx, j, RANGE] # current range
                    primary_rejects_log[curr_reject_idx, C_END]     = taxi_bank[pickup_idx, j, END]   # taxi's shift end
                    primary_rejects_log[curr_reject_idx, C_TG]      = tg_distance                     # requested distance
                    primary_rejects_log[curr_reject_idx, C_PU]      = pickup_idx + 1                  # requested pickup location
                    primary_rejects_log[curr_reject_idx, C_TRIP]    = trip_distance                   # trip distance
                    primary_rejects_log[curr_reject_idx, C_IB]      = ib_miles                        # in between trip distance
                    
                    curr_reject_idx += 1

                    if curr_reject_idx >= 3000:
                        raise Exception("Primary logging overflow")

        # log when taxi's reject rides from within themselves
        
        j += 1
    
    return -1, -1, -1, -1, curr_reject_idx



# NUMBA optimization for secondary search
@njit(cache=True)
def secondary_search_numba(
    pickup_idx,
    pickup_time_seconds,
    trip_distance,

    connecting_DOs, 
    taxi_bank, 
    in_between_starts,
    in_between_lengths,
    in_between_miles,
    in_between_seconds,
    secondary_rejects_log #(N, 9) numpy zero array
):
    
    C_TIME=0; C_ID=1; C_CUR_LOC=2; C_RANGE=3; C_END=4; C_TG=5; C_PU=6; C_TRIP=7; C_IB=8

    curr_reject_idx = 0
    
    i = 0
    while connecting_DOs[pickup_idx][i] > 0:

        DO_idx = connecting_DOs[pickup_idx][i] - 1

        # sample ib_miles and ib_dist
        start = in_between_starts[pickup_idx][DO_idx]
        length = in_between_lengths[pickup_idx][DO_idx]

        sample_index = np.random.randint(0, length)

        ib_miles = in_between_miles[start + sample_index]
        ib_time =  in_between_seconds[start + sample_index]

        tg_distance = (trip_distance + ib_miles)
        
        # search through taxi's in location array until reach np.inf
        j = 1 # start after first marker
        while taxi_bank[DO_idx, j, TIME] < gravestone + 1:

            # haven't reached end indication

            if (taxi_bank[DO_idx, j, TIME] <= pickup_time_seconds and
                taxi_bank[DO_idx, j, END] > pickup_time_seconds):
                # the taxi has valid pickup and end shift times.
                # check if it has enough range and log if it rejects from not having enough
                if taxi_bank[DO_idx, j, RANGE]  >= tg_distance:                
                    return DO_idx, j, ib_miles, ib_time, curr_reject_idx

                else:
                    # add info to reject log
                    secondary_rejects_log[curr_reject_idx, C_TIME]    = pickup_time_seconds         # requested pickup time
                    secondary_rejects_log[curr_reject_idx, C_ID]      = taxi_bank[DO_idx, j, ID]    # taxi id
                    secondary_rejects_log[curr_reject_idx, C_CUR_LOC] = DO_idx + 1                  # current location
                    secondary_rejects_log[curr_reject_idx, C_RANGE]   = taxi_bank[DO_idx, j, RANGE] # current range
                    secondary_rejects_log[curr_reject_idx, C_END]     = taxi_bank[DO_idx, j, END]   # taxi's shift end
                    secondary_rejects_log[curr_reject_idx, C_TG]      = tg_distance                 # requested distance
                    secondary_rejects_log[curr_reject_idx, C_PU]      = pickup_idx + 1              # requested pickup location
                    secondary_rejects_log[curr_reject_idx, C_TRIP]    = trip_distance               # trip distance
                    secondary_rejects_log[curr_reject_idx, C_IB]      = ib_miles                    # in between trip distance
                    
                    curr_reject_idx += 1

                    if curr_reject_idx >= 20_000:
                        raise Exception("Secondary logging overflow")
            
            j += 1

        i += 1
    
    return -1, -1, -1, -1, curr_reject_idx



# NUMBA optimization for array cleaning
@njit(cache=True)
def taxi_bank_cleaning_numba(
    current_time_seconds,
    taxi_bank,
    indexes_for_next_taxis,
    total_distances_np_array,
    next_dist_index
):
    
    null_taxi = np.array([np.inf, 0, 0, 0, 0])

    num_locations, num_taxis, _ = taxi_bank.shape

    for l in range(num_locations):

        # change taxi location array

        t = 1
        index_for_next_valid = 1
        # go through all entries from left till get to a null taxi (hasn't been touched yet)
        while taxi_bank[l][t][TIME] < gravestone_plus_one:

            # check if this taxi is a gravestone
            if taxi_bank[l][t][TIME] == gravestone:
                # taxi is a gravestone, so continue without moving over valid
                pass
            
            else:
                # check if this taxi is on shift
                if taxi_bank[l][t][END] > current_time_seconds:

                    # taxi is on shift, copy it over to next valid index
                    taxi_bank[l][index_for_next_valid][:] = taxi_bank[l][t][:]

                    # increment next valid index
                    index_for_next_valid += 1
                
                else:
                    # taxi is off shift, so continue without moving over valid
                    # log it's distance tho
                    total_distances_np_array[next_dist_index] = taxi_bank[l][t][TOTAL]
                    next_dist_index += 1

            t += 1
        
        indexes_for_next_taxis[l] = index_for_next_valid
        
        # for all the indexes past the next valid, change them to null taxi
        next_invalid = index_for_next_valid
        while taxi_bank[l][next_invalid][TIME] < np.inf:
            taxi_bank[l][next_invalid][:] = null_taxi

            next_invalid += 1
    
    return next_dist_index













TIME = 0
RANGE = 1
TOTAL = 2
END   = 3
ID    = 4

gravestone = np.int64(999_999_999_999)
gravestone_plus_one = gravestone + 1

"""
Class for managing the taxi cab fleet

Will store active taxi cabs in an arry for each LocationID
"""
class TaxiFleet:


    """
    Function to initialize taxi cab fleet

    NOTE: all imputs are in seconds or miles

    inputs:
        - taxi_shifts_df
        
                A pandas dataframe of shift start times and start locationIDs. Should include columns: (start_time, start_locationID, and duration)

                NOTE: everything should be based in seconds

        - taxi_range

                Number of range miles for a single taxi

        - trips_df

                dataframe of trips. Only used to get min and max locationIDs

        - sampling_stuff

                dictionary of sampling dictionaries with:
                    - 'values'  : 1d array
                    - 'starts'  : 2d array of start indexes within values
                    - 'lengths' : 2d array of lengths for original cell arrays (num to sample from)
                    - 'min_row' : minimum value of original df index (should be 1 for LocationIDs)
                    - 'min_col' : minimum value of original df column
                    - 'cols'    : column series of original dataframe
                    - 'rows'    : rows seres of original dataframe

        - PU_to_DO_info

                keys:
                    - 'data' list of lists with dropoff connectors to pickup i
                    - 'min_locationID' original minimum location ID (should be 1)
    
    Cab array:

        The cab array will be size (num location IDs, bank size, 4)
            - the last dimension of 4 will store:
                - time available
                - current_range
                - total_driving
                - shift_end

                
    """
    def __init__(self, taxi_shifts_df, taxi_range, trips_df, sampling_stuff, PU_to_DO_info, low_range_threshold=30, fleet_bank_size=20_000):

        # store info for simulating taxis
        self.taxi_range = taxi_range
        self.low_range_threshold = low_range_threshold


        self.setup_shift_info(taxi_shifts_df)

        self.setup_cab_array(trips_df, fleet_bank_size)

        self.setup_sampling_info(sampling_stuff)

        # setup PU_to_DO array
        self.PU_to_DO_array = ragged_to_padded(PU_to_DO_info['data'], 300)


        # logging stuff for inference
        self.primary_rejects_logs = [] # will be a list of numpy arrays
        self.primary_rejects_np_array = np.zeros((3_000, 9))

        self.secondary_rejects_logs = [] # will be a list of numpy arrays
        self.secondary_rejects_np_array = np.zeros((20_000, 9))

        self.taxi_below_threshold = [] # will be a list of tuples


        num_shifts = len(taxi_shifts_df) # should be maximum number of total distances to log
        self.taxi_total_distance_next_index = 0
        self.taxi_total_distance_np_array = np.zeros(num_shifts + 10_000, dtype=np.float64)


        # logging stuff for time optimization
        self.total_trip_processing_time = 0
        self.search_primary_time = 0
        self.search_secondary_time = 0
        self.time_spawning = 0
        self.post_find_time = 0

        self.search_primary_count = 0
        self.search_secondary_count = 0
        self.search_fails = 0


    """
    Function to setup info for creating taxi drivers and shifts

    input:
        - taxis_shifts_df
                pandas Dataframe with columns: 'start_seconds', 'start_locationID', 'shift_duration'
    """
    def setup_shift_info(self, taxi_shifts_df : pd.DataFrame):
        # store shifts dataframe for use in spawning taxis
        #print("Sorting shift dataframe:")
        taxi_shifts_df = taxi_shifts_df.sort_values(by="start_time")
        self.shifts_df = taxi_shifts_df
        
        # create row indexer for taxi_shift_dataframe to keep track of the next shift to be added
        self.next_shift_row = 0

        # extract shift start_seconds, location ID, and shift duration seconds into numpy arrays for faster looking up
        self.shift_start_times = self.shifts_df['start_time'].to_numpy()
        self.shift_start_location_idxs = self.shifts_df['start_locationID'].to_numpy() - 1 # subtract by 1 since locations start at 1
        self.shift_durations = self.shifts_df['duration'].to_numpy()

        # append np.inf to the end of shift durations to make the final if condition always false (will never spawn taxi past that)
        self.shift_start_times = np.append(self.shift_start_times, np.inf)

    """
    Function to setup the empty taxi cab bank. An empty entry is denoted as [np.inf, 0, 0, 0, 0]
    """
    def setup_cab_array(self, trips_df, fleet_bank_size):

        min_locationID = min(trips_df['PU_LocationID'].min(), trips_df['DO_LocationID'].min())
        max_locationID = max(trips_df['PU_LocationID'].max(), trips_df['DO_LocationID'].max())

        # create empty LocationID taxi cab arrays
        self.location_range_size = max_locationID - min_locationID + 1 # be inclusive
        
        # create the taxi bank numpy array
        self.taxi_bank = np.zeros((self.location_range_size, fleet_bank_size, 5))

        # set all times available to be np.inf to represent gravestone taxis
        self.taxi_bank[:, :, TIME] = np.inf

        # create indexing array for tracking where the next taxi should be added
        self.next_location_taxi_index = np.ones(self.location_range_size, dtype=np.int64)  # set to ones to avoid overwriting beginning flags


    """
    Function to setup stuff for sampling distributions

    Each sampling info dictionary includes:
        - 'values'  : 1d array
        - 'starts'  : 2d array of start indexes within values
        - 'lengths' : 2d array of lengths for original cell arrays (num to sample from)
        - 'min_row' : minimum value of original df index (should be 1 for LocationIDs)
        - 'min_col' : minimum value of original df column
        - 'cols'    : column series of original dataframe
        - 'rows'    : rows seres of original dataframe
    """
    def setup_sampling_info(self, sampling_stuff):

        # load in sampling stuff for in_between trips

        # check that starts and lengths arrays are the same
        assert np.array_equal(sampling_stuff['in_between_miles']['starts'], sampling_stuff['in_between_durations']['starts']), "Error: sampling in_between miles and durations have mismatching starts arrays"
        assert np.array_equal(sampling_stuff['in_between_miles']['lengths'], sampling_stuff['in_between_durations']['lengths']), "Error: sampling in_between miles and durations have mismatching lengths arrays"
        
        # check that columns are dropoff and rows are pickup
        assert sampling_stuff['in_between_durations']['cols'].name == "DOLocationID", "Error: sampling in_between columns are not dropoff locations"
        assert sampling_stuff['in_between_durations']['rows'].name == "PULocationID", "Error: sampling in_between columns are not pickup locations"

        # check that indexing offset is the same
        assert sampling_stuff['in_between_miles']['min_col'] == sampling_stuff['in_between_miles']['min_row'] == sampling_stuff['in_between_durations']['min_col'] == sampling_stuff['in_between_durations']['min_row'], "Error: sampling in_between miles and durations have mismatching min cols or rows"

        self.in_between_starts = sampling_stuff['in_between_miles']['starts']
        self.in_between_lengths = sampling_stuff['in_between_miles']['lengths']
        self.in_between_offset = sampling_stuff['in_between_miles']['min_col']

        self.in_between_miles = sampling_stuff['in_between_miles']['values']
        self.in_between_seconds = sampling_stuff['in_between_durations']['values']


    """
    function to remove all taxi cabs with end times past curr_time

    This has been swithced to the numba optimized function

    """
    def clean_cab_array(self, curr_time_seconds):

        self.taxi_total_distance_next_index = taxi_bank_cleaning_numba(current_time_seconds = curr_time_seconds,
                                                                       taxi_bank = self.taxi_bank,
                                                                       indexes_for_next_taxis=self.next_location_taxi_index,
                                                                       total_distances_np_array=self.taxi_total_distance_np_array,
                                                                       next_dist_index=self.taxi_total_distance_next_index)        

    

    

    def check_spawn_taxi(self, curr_time_seconds):

        # check that the row for the next shift is at or before curr_time
        if self.shift_start_times[self.next_shift_row] <= curr_time_seconds:
            
            i = self.next_shift_row
            j = np.searchsorted(self.shift_start_times, curr_time_seconds, side="right")
            
            for idx in range(i,j):
                self.spawn_taxi(shift_index = idx)

            self.next_shift_row = j
    

    """
    Add a taxi into the taxi array at the location dimension at the index specified in the next taxi index array

    Taxi = [time_available, range, total, end, id]
    """
    def spawn_taxi(self, shift_index):

        # get the location
        loc_idx = self.shift_start_location_idxs[shift_index]

        # get the index for the next taxi at this location
        taxi_idx = self.next_location_taxi_index[loc_idx]



        # slice the taxi info into the taxi bank
        self.taxi_bank[loc_idx, taxi_idx] = (self.shift_start_times[shift_index], # time available at shift start
                                             self.taxi_range, 
                                             0, 
                                             self.shift_start_times[shift_index] + self.shift_durations[shift_index],
                                             shift_index) #use shift index as unique shift id

        # Increment the indexing to add the next taxi at
        self.next_location_taxi_index[loc_idx] += 1

    
    """
    Function that finds a taxi cab for a given trip

    Starts by searching taxis in same pickup location area

    moves to search taxis in other valid areas. These are determined by area pairs that meet a threshold for number of in_between samples from 2013 dataset


    NOTE: This only samples distance once per dropoff location.
    
    I could consider implementing finding all matches then return cab with minimum distance in future


    returns:
        matched dropoff index, index of the taxi within the dropoff, sampled in between distance, sampled in between duration
    """
    def find_taxi_for_trip(self, pickup_time_seconds, pickup_idx, distance):

        primary_time_start = time.time()

        # returns -1 idx when not found
        DO_idx, match_idx, ib_dist, ib_time, num_rejects = primary_search_numba(pickup_idx=pickup_idx,
                                                                                pickup_time_seconds=pickup_time_seconds,
                                                                                trip_distance=distance,
                                                                                taxi_bank= self.taxi_bank, 
                                                                                in_between_starts= self.in_between_starts,
                                                                                in_between_lengths= self.in_between_lengths,
                                                                                in_between_miles= self.in_between_miles,
                                                                                in_between_seconds= self.in_between_seconds,
                                                                                primary_rejects_log=self.primary_rejects_np_array)
        
        # if there are primary range rejection logs, add then to log numpy array list
        if num_rejects != 0:

            # there exists some reject logs, slice the rest and add the numpy array to the list
            self.primary_rejects_logs.append(self.primary_rejects_np_array[:num_rejects].copy())  # slice to index of what would be the next reject

                                                   
        self.search_primary_time += time.time() - primary_time_start
        
        if DO_idx > -1:
            self.search_primary_count += 1
            return DO_idx, match_idx, ib_dist, ib_time 
        

        # PRIMARY SEARCH FAILED, beginning full secondary search
        
        secondary_time_start = time.time()

        # go through each cab within connecting locations and check if valid
        # the location IDs should be sorted by number of connection instances in main dataset
        DO_idx, match_idx, ib_dist, ib_time, num_rejects = secondary_search_numba(pickup_idx=pickup_idx,
                                                                                  pickup_time_seconds= pickup_time_seconds,
                                                                                  trip_distance= distance,
                                                                                  connecting_DOs = self.PU_to_DO_array, 
                                                                                  taxi_bank = self.taxi_bank, 
                                                                                  in_between_starts= self.in_between_starts,
                                                                                  in_between_lengths= self.in_between_lengths,
                                                                                  in_between_miles= self.in_between_miles,
                                                                                  in_between_seconds= self.in_between_seconds,
                                                                                  secondary_rejects_log=self.secondary_rejects_np_array)

        # if there are secondary range rejection logs, add then to log numpy array list
        if num_rejects != 0:

            # there exists some reject logs, slice the rest and add the numpy array to the list
            self.secondary_rejects_logs.append(self.secondary_rejects_np_array[:num_rejects].copy())  # slice to index of what would be the next reject

        self.search_secondary_time += time.time() - secondary_time_start
        
        if DO_idx > -1:
            self.search_secondary_count += 1
            return DO_idx, match_idx, ib_dist, ib_time 
        
        self.search_fails += 1
        return -1, -1, None, None # no taxi found

                    
        
    
    def assign_trip(self, pickup_time_seconds, pickup_location, duration, distance, dropoff_location):

        assign_start = time.time()

        pickup_idx, dropoff_idx = pickup_location - 1, dropoff_location - 1
        
        spawn_time_start = time.time()
        # check if need to spawn more taxis
        self.check_spawn_taxi(pickup_time_seconds)
        self.time_spawning += time.time() - spawn_time_start

        
        # find taxi for a trip
        taxi_list_idx, taxi_idx, ib_dist, ib_time = self.find_taxi_for_trip(pickup_time_seconds, pickup_idx, distance)

        # ib_dist and ib_time are the distance and time for the in_between trip
        # this is when the taxi is driving from its last dropoff to the current pickup

        # return false if no taxi is found
        if taxi_list_idx == -1:
            self.total_trip_processing_time += time.time() - assign_start
            return False
        

        
        cab_post_processing_time_start = time.time()


        # extract taxi info 
        time_available, curr_range, curr_total, shift_end, taxi_id = self.taxi_bank[taxi_list_idx, taxi_idx] 

        # remove taxi from the cab array at its location by setting it's time available to the gravestone
        self.taxi_bank[taxi_list_idx, taxi_idx, TIME] = gravestone


        # when a person is picked up, the taxi could have either already been on the way
        # or the taxi could have just finished, so choose maximum time of the two scenarios
        actual_pickup_time = max(pickup_time_seconds, time_available + ib_time)
        
        new_time_available = actual_pickup_time + duration

        drive_dist = distance + ib_dist
        new_range = curr_range - drive_dist
        new_total = curr_total + drive_dist

        
        # get remaining fuel at end of trip
        # log if it's below a threshold (time, dropoff location ID, and remaining fuel)
        if new_range <= self.low_range_threshold:

            self.taxi_below_threshold.append((new_time_available,  # current time (after trip)
                                              taxi_id,             # taxi id
                                              dropoff_idx + 1,        # current location (after dropoff)
                                              new_range,                            # current range
                                              shift_end,                             # taxi's shift end
                                              distance + ib_dist,  # distance just traveled
                                              distance,            # trip distance just traveled
                                              ib_dist))            # in-between distance just traveled


        # delete the taxi cab if its next available time is past its shift
        if new_time_available >= shift_end:
            # taxi is past shift, so don't add back
            # log it's total distance tho
            self.taxi_total_distance_np_array[self.taxi_total_distance_next_index] = new_total

            self.taxi_total_distance_next_index += 1
            

        else:
            # otherwise, slice it onto the dropoff location bank
            self.taxi_bank[dropoff_idx, self.next_location_taxi_index[dropoff_idx]] = (new_time_available,
                                                                                       new_range, 
                                                                                       new_total, 
                                                                                       shift_end,
                                                                                       taxi_id)

            # increment location latest indexer by 1
            self.next_location_taxi_index[dropoff_idx] += 1
        

        self.post_find_time += time.time() - cab_post_processing_time_start

        # return True since trip was taken
        self.total_trip_processing_time += time.time() - assign_start
        return True

# For doing back logged trips, we can just process them at the time of the next trip and just check if they match a taxi within a set time limit, and if they don't, we can remove them if the current time is also beyond the time limit










def main(test_folder_arg):
    
    # Directory *where this script is located*
    script_dir = Path(__file__).resolve().parent
    # Treat the argument as relative to the script dir
    test_folder = (script_dir / test_folder_arg).resolve()
    

    if not test_folder.is_dir():
        raise FileNotFoundError(f"Test folder {test_folder} doesn't exist!")
    
    # check paths special to test (should just be trip and config)
    shift_df_path = test_folder / "shift_information.parquet"
    test_config_path = test_folder / "test_config.yaml"

    if not shift_df_path.exists():
        raise FileNotFoundError(f"Shift file {shift_df_path} doesn't exist within test folder!")
    if not test_config_path.exists():
        raise FileNotFoundError(f"Config file {test_config_path} doesn't exist within test folder!")

    
    # Create paths for exports
    sim_time_export               = test_folder / "gen_info.txt"
    primary_rejects_export_path   = test_folder / "primary_rejects.parquet"
    secondary_rejects_export_path = test_folder / "secondary_rejects.parquet"
    below_thresholds_export_path  = test_folder / "below_thresholds.parquet"
    unfilled_trips_export_path    = test_folder / "unfilled_trips.parquet"
    total_dist_hist_export        = test_folder / "total_distances_hist.jpg"
    total_dist_numpy_export       = test_folder / "total_distances_array.npy"



    # read test config file
    with open(test_config_path,'r') as f:
        config_dict = yaml.safe_load(f)



    
    # load sampling distributions
    sampling_stuff = load_sampling_stuff()

    # load information for DO to PU connections (filtered by occurance minimum threshold, default is 365)
    PU_to_DO_info = get_PU_to_DO_connections(threshold=config_dict['connections_threshold'])

    
    # load in raw trips
    unformatted_trips = pd.read_parquet("data/sim_info/ground_truth_trips_raw.parquet")
    unformatted_trips = unformatted_trips.sort_values(by="PU_time")

    
    # Convert trip times into seconds since start of 2013
    t0 = pd.Timestamp("2013-01-01 00:00:00")

    trips = unformatted_trips.copy()


    # if there's a set simulation cutoff, then slice trips dataframe
    if config_dict['end_date']:
        end_date = pd.Timestamp(config_dict['end_date'])
        print("Simulation cutoff:", end_date)

        trips = trips[trips['PU_time'] <= end_date]


    # Seconds since start of 2013
    trips['PU'] = (trips['PU_time'] - t0).dt.total_seconds().astype('int')
    trips['DO'] = (trips['DO_time'] - t0).dt.total_seconds().astype('int')
    trips.drop(columns=['PU_time', 'DO_time'], inplace=True)

    # create duration column in seconds and then drop dropoff time
    trips['duration'] = trips['DO'] - trips['PU']
    trips.drop(columns=['DO'], inplace=True)
    trips = trips.sort_values(by='PU')

    # reset trips index (used for flag to clean taxi cab array)
    trips = trips.reset_index()
    trips.drop(columns=['index'], inplace=True)

    
    # load in test trip df
    shift_df = pd.read_parquet(shift_df_path)

    # convert start time to seconds from start of 2013
    shift_df['start_time'] = (shift_df['start_time']  - t0).dt.total_seconds().astype('int')
    # convert duration to seconds (is currently hours)
    shift_df['duration'] = (shift_df['duration'] * 3600).round().astype('int64')



    


    # create fleet with loaded data and config settings
    fleet = TaxiFleet(
        taxi_shifts_df=shift_df, 
        taxi_range=config_dict['taxi_range'], 
        trips_df=trips, 
        sampling_stuff=sampling_stuff, 
        PU_to_DO_info=PU_to_DO_info,
        low_range_threshold=config_dict['low_range_threshold']
    )



    # run through all trips
    start = time.time()

    cab_array_cleaning_interval = config_dict['cab_array_cleaning_interval']

    takens = []
    unfilled_trips = []
    for row in tqdm(trips.itertuples(index=True), total=len(trips)):
        
        taken = fleet.assign_trip(
            pickup_time_seconds = row.PU, 
            pickup_location     = row.PU_LocationID, 
            duration            = row.duration, 
            distance            = row.distance, 
            dropoff_location    = row.DO_LocationID
        )

        takens.append(taken)

        if row.Index % cab_array_cleaning_interval == 0 and row.Index > 100:
            #before_size = np.array([len(taxi_array) for taxi_array in fleet.cab_array]).sum()
            #print("before_size:", before_size)
            fleet.clean_cab_array(curr_time_seconds=row.PU)
            #after_size = np.array([len(taxi_array) for taxi_array in fleet.cab_array]).sum()
            #print("after_size:", after_size)
            #break

        if not taken:
            unfilled_trips.append(row)

    end = time.time()




    print(f"Simulation finished in {end - start} seconds.")
    print("\n\n")
    print("total_trip_processing_time:", fleet.total_trip_processing_time)
    print("search_primary_time:", fleet.search_primary_time)
    print("search_secondary_time:", fleet.search_secondary_time)
    print("time_spawning", fleet.time_spawning)
    print("post_find_time", fleet.post_find_time)
    print()
    print("primary searches:", fleet.search_primary_count)
    print("secondary searches:", fleet.search_secondary_count)
    print("failed searches:", fleet.search_fails)
    print("\n\n")

    # convert data
    takens = np.array(takens)

    primary_rejects_df = None
    if len(fleet.primary_rejects_logs) > 0:
        primary_rejects_df = pd.DataFrame(np.vstack(fleet.primary_rejects_logs), columns=['request_time', 'id', 'location', 'range', 'shift_end', 'requested_distance', 'requested_location', 'requested_trip_distance', 'in_between_distance'])
    else:
        primary_rejects_df = pd.DataFrame(columns=['request_time', 'id', 'location', 'range', 'shift_end', 'requested_distance', 'requested_location', 'requested_trip_distance', 'in_between_distance'])
    
    secondary_rejects_df = None
    if len(fleet.secondary_rejects_logs) > 0:
        secondary_rejects_df = pd.DataFrame(np.vstack(fleet.secondary_rejects_logs), columns=['request_time', 'id', 'location', 'range', 'shift_end', 'requested_distance', 'requested_location', 'requested_trip_distance', 'in_between_distance'])
    else:
        secondary_rejects_df = pd.DataFrame(columns=['request_time', 'id', 'location', 'range', 'shift_end', 'requested_distance', 'requested_location', 'requested_trip_distance', 'in_between_distance'])
    
    
    below_thresholds_df =  pd.DataFrame(fleet.taxi_below_threshold, columns=['request_time', 'id', 'location', 'range', 'shift_end', 'just_traveled', 'just_traveled_trip_distance', 'just_traveled_in_between_distance'])
    unfilled_trips_df = pd.DataFrame(unfilled_trips)

    # revert request time and shift end back into date_times
    if len(primary_rejects_df) > 0:
        primary_rejects_df['request_time'] = pd.to_datetime(primary_rejects_df['request_time'], unit='s', origin='2013-01-01')
        primary_rejects_df['shift_end'] = pd.to_datetime(primary_rejects_df['shift_end'], unit='s', origin='2013-01-01')
    
    if len(secondary_rejects_df) > 0:
        secondary_rejects_df['request_time'] = pd.to_datetime(secondary_rejects_df['request_time'], unit='s', origin='2013-01-01')
        secondary_rejects_df['shift_end'] = pd.to_datetime(secondary_rejects_df['shift_end'], unit='s', origin='2013-01-01')



    if len(below_thresholds_df) > 0:
        below_thresholds_df['request_time'] = pd.to_datetime(below_thresholds_df['request_time'], unit='s', origin='2013-01-01')
        below_thresholds_df['shift_end'] = pd.to_datetime(below_thresholds_df['shift_end'], unit='s', origin='2013-01-01')


    # convert unfilled trips to datetime pickup and hour duration
    if len(unfilled_trips_df) > 0:
        unfilled_trips_df['PU'] = pd.to_datetime(unfilled_trips_df['PU'], unit='s', origin='2013-01-01')
        unfilled_trips_df['duration'] = unfilled_trips_df['duration'] / 3600


    # slice total distances array and make a histogram
    total_dists = fleet.taxi_total_distance_np_array[:fleet.taxi_total_distance_next_index]

    # create histogram
    plt.hist(total_dists, bins=200)
    plt.xlabel("total miles driven")
    plt.ylabel("density")
    plt.title(f"Histogram of total miles for {len(total_dists):,} taxis")
    plt.savefig(total_dist_hist_export, dpi=300, bbox_inches='tight')
    plt.close()


    # save information
    print("Saving info...")
    primary_rejects_df.to_parquet(primary_rejects_export_path, index=False)
    secondary_rejects_df.to_parquet(secondary_rejects_export_path, index=False)
    below_thresholds_df.to_parquet(below_thresholds_export_path, index=False)
    unfilled_trips_df.to_parquet(unfilled_trips_export_path, index=False)
    np.save(total_dist_numpy_export, total_dists)

    print("done")
    print("\n\n")

    # print takens stats
    print("number of non filled rides:", (takens == 0).sum())
    print("percentage:", takens.mean())


    # write to save file
    with open(sim_time_export, "w") as f:
        f.write(f"Simulation finished in {end - start} seconds.\n")
        f.write(f"number of non filled rides: {(takens == 0).sum()}\n")
        f.write(f"percentage: {takens.mean()}\n")








if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python simulation_v1.py test_folder")
        print("NOTE: all configs done in test_config.yaml within test folder")
        sys.exit(1)

    argument = sys.argv[1]
    main(argument)

















