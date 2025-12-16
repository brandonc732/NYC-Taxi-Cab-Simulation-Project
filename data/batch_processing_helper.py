import pandas as pd

def process_saved_batch(batch_num):

    print("Processing batch:", batch_num)
    
    batch_df = pd.read_parquet(f"distilled taxi data/between trip batches/batch_{batch_num}/batch_df.parquet")

    # Important: sort by license + time so shift(-1) stays within each license
    batch_df = batch_df.sort_values(["hack_license", "dropoff_datetime"])

    licenses = batch_df['hack_license'].unique()

    cols = ['hack_license', 'dropoff_datetime', 'dropoff_longitude', 'dropoff_latitude', 'DOLocationID_2013', 'next_pickup_time', 'next_pickup_latitude', 'next_pickup_longitude', 'next_PULocationID']
    
    batch_result = []

    for license in licenses:
        sub_df = batch_df[batch_df['hack_license'] == license].copy()
    
        sub_df = sub_df.sort_values(['dropoff_datetime'])
    
        sub_df["next_pickup_time"]      = sub_df['pickup_datetime'].shift(-1)
        sub_df["next_pickup_latitude"]  = sub_df['pickup_latitude'].shift(-1)
        sub_df["next_pickup_longitude"] = sub_df['pickup_longitude'].shift(-1)
        sub_df["next_PULocationID"]     = sub_df['PULocationID_2013'].shift(-1)
    
        between_trips = sub_df[cols]
    
        batch_result.append(between_trips)

    del sub_df
    del batch_df

    print("Finished loop for batch", batch_num)

    batch_result = pd.concat(batch_result, ignore_index=True)

    print(f"NAs for batch {batch_num}:", batch_result.isna().sum())
    
    
    print("saving results for batch :", batch_num)
    
    out_path = f"distilled taxi data/between trip batches/batch_{batch_num}/batch_result.parquet"
    batch_result.to_parquet(out_path, index=False)

    result_info = (batch_num, len(batch_result))

    del batch_result

    return result_info
    
    
    
    
    
    
    
    
    
    
    
    
"""
    # Compute "next_*" within each license
    g = batch_df.groupby("hack_license", sort=False)
    batch_df["next_pickup_time"]      = g["pickup_datetime"].shift(-1)
    batch_df["next_pickup_latitude"]  = g["pickup_latitude"].shift(-1)
    batch_df["next_pickup_longitude"] = g["pickup_longitude"].shift(-1)
    batch_df["next_PULocationID"]     = g["PULocationID_2013"].shift(-1)

    cols = ['hack_license', 'dropoff_datetime', 'dropoff_longitude', 'dropoff_latitude', 'DOLocationID_2013', 'next_pickup_time', 'next_pickup_latitude', 'next_pickup_longitude', 'next_PULocationID']

    print(f"NAs for batch {batch_num}:", batch_df.isna().sum())
    
    between_trips = batch_df.dropna()[cols]

    
    print("saving results for batch :", batch_num)
    
    out_path = f"distilled taxi data/between trip batches/batch_{batch_num}/batch_result.parquet"
    between_trips.to_parquet(out_path, index=False)

    return (batch_num, len(between_trips))
"""