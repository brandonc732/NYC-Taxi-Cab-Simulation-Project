import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import time
import yaml

import os
import sys
from pathlib import Path



"""
Function to convert 2D dataframe of numpy arrays to the following:
- large 1D array of values
- 2D array of bin starts within the 1D array
- 2D array of bin lengths within the 1D array
- If a cell entry is an empty array, then it will be replaced by one entry of np.inf (should only apply to inbetween variables)

"""
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





"""
Class for simulating a NYC taxi driver

attributes:

"""
class Taxi:
    __slots__ = ("_range", "_id", "LocationID", "shift_end", "time_available")
    def __init__(self, shift_start_time, shift_start_locationID, shift_duration, _range, _id):

        self._range = _range
        self._id = _id
        
        self.LocationID = shift_start_locationID
        self.shift_end = shift_start_time + shift_duration

        self.time_available = shift_start_time 










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

                
    """
    def __init__(self, taxi_shifts_df, taxi_range, trips_df, sampling_stuff, PU_to_DO_info, low_range_threshold=30):

        # store info for simulating taxis
        self.taxi_range = taxi_range
        self.low_range_threshold = low_range_threshold


        self.setup_shift_info(taxi_shifts_df)

        self.setup_cab_array(trips_df)

        self.setup_sampling_info(sampling_stuff)

        # save PU_to_DO list
        self.PU_to_DO = PU_to_DO_info['data']
        self.PU_to_DO_offset = PU_to_DO_info['min_locationID']


        # logging stuff for debugging
        self.cab_array_size_log = []

        self.rng = np.random.default_rng()


        # logging stuff for inference
        self.taxi_rejections = []      # will be a list of dictionaries
        self.taxi_below_threshold = [] # will be a list of dictionaries
        


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
        self.shift_start_locations = self.shifts_df['start_locationID'].to_numpy()
        self.shift_durations = self.shifts_df['duration'].to_numpy()

        # append np.inf to the end of shift durations to make the final if condition always false (will never spawn taxi past that)
        self.shift_start_times = np.append(self.shift_start_times, np.inf)

    """
    Function to setup the empty cab lists and offset for indexing it
    """
    def setup_cab_array(self, trips_df):

        min_locationID = min(trips_df['PU_LocationID'].min(), trips_df['DO_LocationID'].min())
        max_locationID = max(trips_df['PU_LocationID'].max(), trips_df['DO_LocationID'].max())

        # create empty LocationID taxi cab arrays
        self.cab_indexing_offset = min_locationID
        self.cab_array = [[] for _ in range(min_locationID, max_locationID + 1)]


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


    def log_cab_array_size(self):
        
        curr_log = [len(taxi_list) for taxi_list in self.cab_array]
        
        self.cab_array_size_log.append(curr_log)


    """
    function to remove all taxi cabs with end times past curr_time
    """
    def clean_cab_array(self, curr_time):

        #print("cleaning taxi array")
        for i, taxi_list in enumerate(self.cab_array):
            if taxi_list:
                taxi_list[:] = [t for t in taxi_list if t.shift_end >= curr_time]
        #print("done")
        

    """
    Sample in between trip from starting location to ending location

    based on sampling_stuff function, the columns=DOLocationID (startin_location), the rows=PULocationID(ending_location)
    """
    def sample_in_between_trip(self, starting_location, ending_location):

        ending_index, starting_idx = ending_location - self.in_between_offset, starting_location- self.in_between_offset
        
        start = self.in_between_starts[ending_index, starting_idx]
        length = self.in_between_lengths[ending_index, starting_idx]

        sample_index = self.rng.integers(length)
        
        miles = self.in_between_miles[start + sample_index]
        seconds = self.in_between_seconds[start + sample_index]

        return miles, seconds

    

    def check_spawn_taxi(self, curr_time_seconds):

        # check that the row for the next shift is at or before curr_time
        """
        while self.shift_start_times[self.next_shift_row] <= curr_time_seconds:

            self.spawn_taxi(shift_index = self.next_shift_row)

            self.next_shift_row += 1
        """
        
        # check that the row for the next shift is at or before curr_time
        if self.shift_start_times[self.next_shift_row] <= curr_time_seconds:
            
            i = self.next_shift_row
            j = np.searchsorted(self.shift_start_times, curr_time_seconds, side="right")
            
            for idx in range(i,j):
                self.spawn_taxi(shift_index = idx)

            self.next_shift_row = j
    

    def spawn_taxi(self, shift_index):

        # spawn a taxi based on the shift data at shift_index
        new_taxi = Taxi(shift_start_time =       self.shift_start_times[shift_index], 
                        shift_start_locationID = self.shift_start_locations[shift_index], 
                        shift_duration =         self.shift_durations[shift_index], 
                        _range =                 self.taxi_range,
                        _id = shift_index)

        # add new taxi into corresponding locationID array
        self.cab_array[new_taxi.LocationID - self.cab_indexing_offset].append(new_taxi)

    
    """
    Function that finds a taxi cab for a given trip

    Starts by searching taxis in same pickup location area

    moves to search taxis in other valid areas. These are determined by area pairs that meet a threshold for number of in_between samples from 2013 dataset


    NOTE: This only samples distance once per dropoff location.
    
    I could consider implementing finding all matches then return cab with minimum distance in future


    returns:
        matched taxi object, index of its cab array (dropoffID), its index within that array
    """
    def find_taxi_for_trip(self, pickup_time_seconds, pickup_location, distance):

        # get cab list at pickup_location
        taxi_list = self.cab_array[pickup_location - self.cab_indexing_offset]

        # go through each cab within the same loction and check if valid
        last_dropoff_location = pickup_location
        ib_dist, ib_time = self.sample_in_between_trip(starting_location=last_dropoff_location,
                                                         ending_location=pickup_location)

        target_distance = distance + ib_dist
        
        for i in range(len(taxi_list)):
            # check if time available is before pickup time
            if taxi_list[i].time_available <= pickup_time_seconds:

                # check if it's still on shift
                if taxi_list[i].shift_end > pickup_time_seconds:
                    
                    # check if it has enough range
                    if taxi_list[i]._range >= target_distance:
                        return taxi_list[i], pickup_location, i, ib_dist, ib_time

                    else:
                        # log rejected taxi cause of range
                        #print("Taxi skipped for no range")
                        self.taxi_rejections.append((pickup_time_seconds,      # requested pickup time
                                                     taxi_list[i]._id,         # taxi id
                                                     taxi_list[i].LocationID,  # current location
                                                     taxi_list[i]._range,      # current range
                                                     taxi_list[i].shift_end,   # taxi's shift end
                                                     target_distance,          # requested distance
                                                     pickup_location,          # pickup location
                                                     distance,                 # trip distance
                                                     ib_dist))                  # in-between distance
                                                   

        
        # go through each cab within connecting locations and check if valid
        # the location IDs should be sorted by number of connection instances in main dataset
        connecting_DOs = self.PU_to_DO[pickup_location - self.PU_to_DO_offset]

        for DO_location in connecting_DOs:
            # get cab list at connecting dropoff 
            taxi_list = self.cab_array[DO_location - self.cab_indexing_offset]

            ib_dist, ib_time = self.sample_in_between_trip(starting_location=DO_location,
                                                           ending_location=pickup_location)
            
            target_distance = distance + ib_dist
            
            for i in range(len(taxi_list)):
                # check if time available is before pickup time
                if taxi_list[i].time_available <= pickup_time_seconds:
    
                    # check if it's still on shift
                    if taxi_list[i].shift_end > pickup_time_seconds:
                        
                        # check if it has enough range
                        if taxi_list[i]._range >= target_distance:
                            return taxi_list[i], DO_location, i, ib_dist, ib_time
    
                        else:
                            # log rejected taxi cause of range
                            #print("Taxi skipped for no range")
                            self.taxi_rejections.append((pickup_time_seconds,      # requested pickup time
                                                         taxi_list[i]._id,         # taxi id
                                                         taxi_list[i].LocationID,  # current location
                                                         taxi_list[i]._range,      # current range
                                                         taxi_list[i].shift_end,   # taxi's shift end
                                                         target_distance,          # requested distance
                                                         pickup_location,          # pickup location
                                                         distance,                 # trip distance
                                                         ib_dist))                  # in-between distance

        
        return None, None, None, None, None # no taxi found

                    
        
    
    def assign_trip(self, pickup_time_seconds, pickup_location, duration, distance, dropoff_location):
        # check if need to spawn more taxis
        self.check_spawn_taxi(pickup_time_seconds)

        
        # find taxi for a trip
        found = False
        taxi, cab_list_location, taxi_array_index, ib_dist, ib_time = self.find_taxi_for_trip(pickup_time_seconds, pickup_location, distance)

        # ib_dist and ib_time are the distance and time for the in_between trip
        # this is when the taxi is driving from its last dropoff to the current pickup

        # return false if no taxi is found
        if taxi == None:
            return False

        # remove taxi from the cab list at it's location
        # removed_cab = self.cab_array[cab_list_location - self.cab_indexing_offset].pop(taxi_array_index)
        lst = self.cab_array[cab_list_location - self.cab_indexing_offset]
        removed_cab = lst[taxi_array_index]
        lst[taxi_array_index] = lst[-1]
        lst.pop()
        
        # assign the taxi to the trip
        taxi.LocationID = dropoff_location

        # when a person is picked up, the taxi could have either already been on the way
        # or the taxi could have just finished, so choose maximum time of the two scenarios
        actual_pickup_time = max(pickup_time_seconds, taxi.time_available + ib_time)
        
        taxi.time_available = actual_pickup_time + duration
        taxi._range -= (distance + ib_dist)

        
        # get remaining fuel at end of trip
        # log if it's below a threshold (time, dropoff location ID, and remaining fuel)
        if taxi._range <= self.low_range_threshold:
            # log rejected taxi cause of range
            #print("Taxi below range threshold")
            self.taxi_below_threshold.append((pickup_time_seconds, # current time
                                              taxi._id,            # taxi id
                                              taxi.LocationID,     # current location (after dropoff)
                                              taxi._range,         # current range
                                              taxi.shift_end,      # taxi's shift end
                                              distance + ib_dist,  # distance just traveled
                                              distance,            # trip distance just traveled
                                              ib_dist))            # in-between distance just traveled


        # delete the taxi cab if its next available time is past its shift
        if taxi.time_available >= taxi.shift_end:
            del taxi

        else:
            # otherwise, append it onto the taxi cab list at the dropoff location
            self.cab_array[dropoff_location - self.cab_indexing_offset].append(taxi)

        #self.log_cab_array_size()

        # return True since trip was taken
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
    sim_time_export              = test_folder / "gen_info.txt"
    rejects_export_path          = test_folder / "rejects.parquet"
    below_thresholds_export_path = test_folder / "below_thresholds.parquet"
    unfilled_trips_export_path   = test_folder / "unfilled_trips.parquet"


    # read test config file
    with open(test_config_path,'r') as f:
        config_dict = yaml.safe_load(f)



    
    # load sampling distributions
    sampling_stuff = load_sampling_stuff()

    # load information for DO to PU connections (filtered by occurance minimum threshold, default is 365)
    PU_to_DO_info = get_PU_to_DO_connections(threshold=config_dict['connections_threshold'])

    
    # load in raw trips
    unformatted_trips = pd.read_parquet("data/sim_info/sim_trips_raw.parquet")
    unformatted_trips = unformatted_trips.sort_values(by="PU_time")

    
    # Convert trip times into seconds since start of 2023
    t0 = pd.Timestamp("2023-01-01 00:00:00")

    trips = unformatted_trips.copy()

    # Seconds since start of 2023
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

    # convert start time to seconds from start of 2023
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
            fleet.clean_cab_array(curr_time=row.PU)
            #after_size = np.array([len(taxi_array) for taxi_array in fleet.cab_array]).sum()
            #print("after_size:", after_size)
            #break

        if not taken:
            unfilled_trips.append(row)

    end = time.time()




    print(f"Simulation finished in {end - start} seconds.")

    # convert data
    takens = np.array(takens)
    rejects_df = pd.DataFrame(fleet.taxi_rejections, columns=['request_time', 'id', 'location', 'range', 'shift_end', 'requested_distance', 'requested_location', 'requested_trip_distance', 'in_between_distance'])
    below_thresholds_df =  pd.DataFrame(fleet.taxi_below_threshold, columns=['request_time', 'id', 'location', 'range', 'shift_end', 'just_traveled', 'just_traveled_trip_distance', 'just_traveled_in_between_distance'])
    unfilled_trips_df = pd.DataFrame(unfilled_trips)

    # revert request time and shift end back into date_times
    if len(rejects_df) > 0:
        rejects_df['request_time'] = pd.to_datetime(rejects_df['request_time'], unit='s', origin='2023-01-01')
        rejects_df['shift_end'] = pd.to_datetime(rejects_df['shift_end'], unit='s', origin='2023-01-01')

    if len(below_thresholds_df) > 0:
        below_thresholds_df['request_time'] = pd.to_datetime(below_thresholds_df['request_time'], unit='s', origin='2023-01-01')
        below_thresholds_df['shift_end'] = pd.to_datetime(below_thresholds_df['shift_end'], unit='s', origin='2023-01-01')


    # convert unfilled trips to datetime pickup and hour duration
    if len(unfilled_trips_df) > 0:
        unfilled_trips_df['PU'] = pd.to_datetime(unfilled_trips_df['PU'], unit='s', origin='2023-01-01')
        unfilled_trips_df['duration'] = unfilled_trips_df['duration'] / 3600



    # save information
    print("Saving info...")
    rejects_df.to_parquet(rejects_export_path, index=False)
    below_thresholds_df.to_parquet(below_thresholds_export_path, index=False)
    unfilled_trips_df.to_parquet(unfilled_trips_export_path, index=False)

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





































































































































































































































































































