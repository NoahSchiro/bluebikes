import os
import pandas
import sqlite3

# Handles the "normal" case
def handle_normal(df):
    # Drop columns we don't care about
    df.drop(["gender", "birth year", "bikeid"], axis=1, inplace=True)
    print("Dropped columns...")

    return df

# Handles dataframes that end with a postal code
def handle_postal(df):
    df.drop(["postal code", "bikeid"], axis=1, inplace=True)
    print("Dropped columns...")
    return df

# Handles dataframes that are completely different from the other ones
def handle_weird(df):
    # Columns we don't care about
    df.drop(["ride_id", "rideable_type"], axis=1, inplace=True)
    print("Dropped columns...")

    # Columns we need to rename
    df.rename({
        "started_at" : "starttime",
        "ended_at" : "stoptime",
        
        "start_station_name" : "start station name",
        "start_station_id" : "start station id",
        "start_lat" : "start station latitude",
        "start_lng" : "start station longitude",
        
        "end_station_name" : "end station name",
        "end_station_id" : "end station id",
        "end_lat" : "end station latitude",
        "end_lng" : "end station longitude",
 
        "member_casual" : "usertype",
    }, axis=1, inplace=True)
    print("Renamed...")

    # Rename the usertype column
    def mapping(input_str):
        if input_str == "member":
            return "Subscriber"
        elif input_str == "casual":
            return "Customer"
        else:
            print(f"error on {input_str}")
            exit(1)

    # Map columns to new names
    df["usertype"].apply(mapping)

    print("Remapped usertype")

    # Convert columns to datetime
    df["starttime"] = pandas.to_datetime(df["starttime"])
    df["stoptime"] = pandas.to_datetime(df["stoptime"])

    # Calculate elapsed seconds
    df["tripduration"] = (df["stoptime"] - df["starttime"]).dt.total_seconds()
    print("computed seconds")

    return df

def handle_df(df):
    first = df.columns.to_list()[0]
    last = df.columns.to_list()[-1]

    if first == "tripduration" and last == "gender":
        print("Identified as normal")
        return handle_normal(df)
    elif first == "tripduration" and last == "postal code":
        print("Identified as postal")
        return handle_postal(df)
    elif first == "ride_id":
        print("Identified as weird")
        return handle_weird(df)
    else:
        print("Did not expect a DF with header: ")
        print(df.columns.to_list())

def collect_csv(filepath="../data/"):

    dfs = []

    for root, _, files in os.walk(filepath):
        for file in files:
            if file.endswith(".csv"):
                print(f"Loading {file}...")
                path = os.path.join(root, file)
                dfs.append(pandas.read_csv(path))

    return dfs

def register(conn, list_of_df):
    # Ensure column order is consistent
    columns = list_of_df[0].columns  # Assume the first DataFrame's column order as the standard
    aligned_dataframes = [df[columns] for df in list_of_df]

    print("Writing dfs to db...", end="")
    # Iterate over DataFrames and append to the SQLite table
    for df in aligned_dataframes:

        # Append to the SQLite table
        df.to_sql("blue bikes", conn, if_exists="append", index=False)
    print(" complete!")

if __name__=="__main__":

    conn = sqlite3.connect("data.db")

    data = collect_csv()
    new_data = [handle_df(df) for df in data]
    register(conn, new_data)
