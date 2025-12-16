import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import time

from pathlib import Path
import os



def remove_outliers_iqr(arr, k=3.0):
    """
    Remove extreme outliers from a 1D numpy array using the IQR rule.
    Returns a new array with only non-outlier values.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr  # keep empty arrays as-is

    # Use percentiles for robustness
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        # All values nearly identical; nothing to remove
        return arr

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask = (arr >= lower) & (arr <= upper)

    removed = arr[~mask]
    if len(removed) > 0:
        #print("Removed:", removed)
        #print("Kepted:", arr[mask])
        #print("\n\n")
        pass
    return arr[mask]


def generate_shifts_dataframe(driver_levels_2013 = 33_300, driver_levels_2023 = 11_700, extra_reducer=0.7):

    # load data
    print("Shift start counts...")
    shift_counts_df = pd.read_pickle("data/sim_info/shift_start_counts_arrays.pkl")
    
    print("Shift durations...")
    shift_durations_df = pd.read_pickle("data/sim_info/shift_arrays.pkl")
    
    print("Shift start locations...")
    shift_start_locationsIDs_df = pd.read_pickle("data/sim_info/shift_start_location_arrays.pkl")
    
    print("driver counts...")
    driver_count_df = pd.read_pickle("data/sim_info/driver_count_arrays.pkl")


    # filter extreme outliers in count dataframes
    shift_counts_df = shift_counts_df.map(remove_outliers_iqr)
    driver_count_df = driver_count_df.map(remove_outliers_iqr)



    # scale the count levels from 2013 to 2023 with extra reducer for shift hours and stuff
    reduction = extra_reducer * (driver_levels_2023 / driver_levels_2013)
    

    # apply scaling
    print("Applying 2013 to 2023 scaling")
    
    shift_counts_df = shift_counts_df.map(lambda arr: np.rint(arr * reduction).astype(np.int16))
    driver_count_df = driver_count_df.map(lambda arr: np.rint(arr * reduction).astype(np.int16))

    
    # calculate mean and std for z-score adjustments
    driver_count_means = driver_count_df.map(np.mean)
    driver_count_stds = driver_count_df.map(np.std)

    



    
    # GENERATION SECTION



    # start a couple days before 2023
    start_time = pd.Timestamp('2022-12-20')
    adjustment_start = start_time + pd.Timedelta(days=5)
    
    # z_scores to add to remove shifts at
    low_z_threshold = -3
    high_z_threshold = 3
    
    
    # end a couple days after 2023
    timestamp_end = pd.Timestamp('2024-01-02')
    #timestamp_end = pd.Timestamp('2023-01-02')
    
    shifts_dataframe = pd.DataFrame(columns=['start_time', 'start_locationID', 'duration', 'end_time'])
    
    
    z_scores_list = []
    z_adjustment_messages = []
    
    hours = pd.date_range(start=start_time, end=timestamp_end, freq="h")
    n = len(hours)
    
    
    # make this long enough for max duration you might see (pick a safe buffer)
    end_counts = np.zeros(n + 72*2, dtype=np.int32)  # e.g. +144 hours buffer
    current_active = 0
    
    chunks = []
    meta = []
    
    for t_idx, curr_time in tqdm(enumerate(hours), total=len(hours)):
    
        current_active -= end_counts[t_idx]
    
        # get hour and day of week for current time
        hour = curr_time.hour
        dow  = curr_time.day_of_week
    
        # sample the number of shifts to generate
        num_starts = np.random.choice(shift_counts_df[dow][hour])
    
        # sample data arrays for these shifts
        new_location_IDs = np.random.choice(shift_start_locationsIDs_df[dow][hour], replace=True, size=num_starts)
        new_durations    = np.random.choice(shift_durations_df[dow][hour], replace=True, size=num_starts)
    
        # create new shifts
        new_shifts = []
        new_end_buckets = []
        for i in range(num_starts):
            # sample a new shift
            # I'm just going to start shifts on the hour
            new_shift_location_ID = new_location_IDs[i]
            new_shift_duration = new_durations[i]
            new_shift_end = curr_time + pd.Timedelta(hours=new_shift_duration)
    
            end_bucket = t_idx + np.ceil(new_shift_duration).astype(np.int32)
            end_counts[end_bucket] += 1
            new_end_buckets.append(end_bucket)
    
            new_shifts.append({
                'start_time' : curr_time,
                'start_locationID' : new_shift_location_ID,
                'duration' : new_shift_duration,
                'end_time' : new_shift_end
            })
    
        # create new shift dataframe
        new_shift_df = pd.DataFrame(new_shifts)
    
        
        current_active += num_starts
    
        chunks.append(new_shift_df)
        meta.append(np.array(new_end_buckets))    

        # since extreme outliers have been filtered with IQR, I'm going to
        # use standard z-score for outlier detection for generation driver counts
        obs_mean = driver_count_means[dow][hour]
        obs_std  = driver_count_stds[dow][hour]
        z_score = (current_active - obs_mean) / obs_std
    
        if z_score > high_z_threshold and curr_time > adjustment_start: # there's too many drivers
    
            # remove drivers to bring back into range
            desired_amount = obs_mean + high_z_threshold * obs_std
    
            amount_to_remove = int(current_active - desired_amount)
    
            log_message = f"{curr_time.strftime('%Y-%m-%d')}   dow: {dow},  hour: {hour},   current: {current_active}, amount to remove: {amount_to_remove}"
            #print(log_message)
            z_adjustment_messages.append(log_message)

            # remove rows from previous chunks and current_active count
            k = amount_to_remove
            while k > 0 and chunks:
                last_df = chunks[-1]
                last_buckets = meta[-1]
                take = min(k, len(last_df))
    
                # undo those shifts in the end_counts + active count
                np.add.at(end_counts, last_buckets[-take:], -1)
                current_active -= take
    
                # trim/pop chunk
                if take == len(last_df):
                    chunks.pop(); meta.pop()
                else:
                    chunks[-1] = last_df.iloc[:-take]
                    meta[-1] = last_buckets[:-take]
                k -= take
    
            
    
    
        if z_score < low_z_threshold and curr_time > adjustment_start: # theres too little drivers
    
            # remove drivers to bring back into range
            desired_amount = obs_mean + low_z_threshold * obs_std
    
            amount_to_add = int(desired_amount - current_active)
    
            log_message = f"{curr_time.strftime('%Y-%m-%d')}   dow: {dow},  hour: {hour},   current: {current_active}, amount to add: {amount_to_add}"
            #print(log_message)
            z_adjustment_messages.append(log_message)
    
            
            # create that many drivers
            new_location_IDs = np.random.choice(shift_start_locationsIDs_df[dow][hour], replace=True, size=amount_to_add)
            new_durations    = np.random.choice(shift_durations_df[dow][hour], replace=True, size=amount_to_add)
    
            # create new shifts
            new_shifts = []
            new_end_buckets = []
            for i in range(amount_to_add):
                # sample a new shift
                # I'm just going to start shifts on the hour
                new_shift_location_ID = new_location_IDs[i]
                new_shift_duration = new_durations[i]
                new_shift_end = curr_time + pd.Timedelta(hours=new_shift_duration)
        
                end_bucket = t_idx + np.ceil(new_shift_duration).astype(np.int32)
                end_counts[end_bucket] += 1
                new_end_buckets.append(end_bucket)
        
                new_shifts.append({
                    'start_time' : curr_time,
                    'start_locationID' : new_shift_location_ID,
                    'duration' : new_shift_duration,
                    'end_time' : new_shift_end
                })
        
            # create new shift dataframe
            new_shift_df = pd.DataFrame(new_shifts)
        
            
            current_active += amount_to_add
        
            chunks[-1] = pd.concat([chunks[-1], new_shift_df], ignore_index=True)
            meta[-1] = np.concat((meta[-1], np.array(new_end_buckets)))
    
    
        z_scores_list.append(z_score)


    print("Creating dataframe:...")
    z_scores = np.array(z_scores_list)
    shift_dataframe = pd.concat(chunks)
    print("done")


    # return dataframe and z_scores. Filtering is done in other loop
    return shift_dataframe, z_scores
    


def create_test_shift_information(test_folder, make_folder):

    # Directory *where this script is located*
    #script_dir = Path(__file__).resolve().parent
    # Treat the argument as relative to the script dir
    test_folder = Path(test_folder)

    #print(test_folder)


    if make_folder:
        # check that the test folder does not exist
        if test_folder.is_dir():
            raise FileNotFoundError(f"Test folder {test_folder} exists while make_folder is True!")

        os.mkdir(test_folder)

    else:
        # check that test folder exists
        if not test_folder.is_dir():
            raise FileNotFoundError(f"Test folder {test_folder} doesn't exist!")

    # make export files
    shift_dataframe_export_path  = test_folder / "shift_information.parquet"
    z_scores_plot_export         = test_folder / "zscores_plot.jpg"
    z_scores_numpy_export        = test_folder / "below_thresholds.npy"

    
    
    # generate z_scores and shift dataframe
    shift_dataframe, z_scores = generate_shifts_dataframe()



    # format and clean shift dataframe to save
    

    # slice out anything that starts in 2024
    sliced_shifts = shift_dataframe[shift_dataframe['start_time'] < pd.Timestamp('2024-01-01')]
    
    # slice out anything that ends before 2023
    sliced_shifts = sliced_shifts[sliced_shifts['end_time'] >= pd.Timestamp('2023-01-01')]
    
    # clamp start times to begging of 2023 and recalculate duration based on end_time
    cutoff = pd.Timestamp('2023-01-01 00:00:00')
    mask = sliced_shifts['start_time'] < cutoff
    
    # Update duration for affected rows (in hours)
    sliced_shifts.loc[mask, 'duration'] = (
        sliced_shifts.loc[mask, 'end_time'] - cutoff
    ).dt.total_seconds() / 3600
    
    # Clamp start_time to cutoff
    sliced_shifts.loc[mask, 'start_time'] = cutoff


    # set location ID column to int16
    sliced_shifts['start_locationID'] = sliced_shifts['start_locationID'].astype(np.int16)
    # remove end_time (not required for simulation)
    export_shifts = sliced_shifts.drop(columns=['end_time'])

    export_shifts.to_parquet(shift_dataframe_export_path)




    # slice out beginning z_scores, make plot and save
    z_scores = z_scores[24*5:]

    low_z_threshold = -3
    high_z_threshold = 3
    out_of_threshold = ((z_scores < low_z_threshold) | (z_scores > high_z_threshold)).mean() * 100

    
    
    plt.plot(z_scores)
    plt.title(f"driver_count z_score plot, {out_of_threshold}% out of threshold")
    plt.xlabel("hour index")
    plt.ylabel("active_driver_count_z_score")

    plt.savefig(z_scores_plot_export, dpi=300, bbox_inches='tight')
    plt.close()


    np.save(z_scores_numpy_export, z_scores)