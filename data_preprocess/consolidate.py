import os
import pandas
import sqlite3

# Handles the "normal" case
def handle_normal(df):
    # Drop columns we don't care about
    df.drop(["gender", "birth year", "bikeid"], axis=1, inplace=True)
    # Columns we need to rename
    df.rename({
        "start station name" : "start_station_name",
        "start station id" : "start_station_id",
        "start station latitude" : "start_lat",
        "start station longitude" : "start_lng",
        
        "end station name" : "end_station_name",
        "end station id" : "end_station_id",
        "end station latitude" : "end_lat",
        "end station longitude" : "end_lng",
    }, axis=1, inplace=True)

    print("Dropped columns...")
    return df

# Handles dataframes that end with a postal code
def handle_postal(df):
    df.drop(["postal code", "bikeid"], axis=1, inplace=True)
    
    # Columns we need to rename
    df.rename({
        "start station name" : "start_station_name",
        "start station id" : "start_station_id",
        "start station latitude" : "start_lat",
        "start station longitude" : "start_lng",
        
        "end station name" : "end_station_name",
        "end station id" : "end_station_id",
        "end station latitude" : "end_lat",
        "end station longitude" : "end_lng",
    }, axis=1, inplace=True)

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

def main(conn, columns, filepath):

    for root, _, files in os.walk(filepath):
        for file in files:
            if file.endswith(".csv"):
                print(f"Loading {file}...")
                # Collect
                path = os.path.join(root, file)
                data_frame = pandas.read_csv(path)
                # Transform
                data_frame = handle_df(data_frame)
                # Write
                register(conn, columns, data_frame)
                # Free up memory
                del data_frame

def register(conn, columns, df):
    # Ensure column order is consistent
    aligned_dataframe = df[columns]

    print("Writing dfs to db...", end="")
    # Iterate over DataFrames and append to the SQLite table
    
    # Append to the SQLite table
    aligned_dataframe.to_sql("blue_bikes", conn, if_exists="append", index=False)

    print(" complete!")

if __name__=="__main__":
    filepath = input("Data location: ")

    conn = sqlite3.connect(os.path.join(filepath, "data.db"))
    columns = [
        "tripduration",
        "starttime",
        "stoptime",
        "start_station_id",
        "start_station_name",
        "start_lat",
        "start_lng",
        "end_station_id",
        "end_station_name",
        "end_lat",
        "end_lng",
        "usertype",
    ]

    # Create a table with an auto-incrementing primary key
    conn.execute("""
    CREATE TABLE IF NOT EXISTS blue_bikes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tripduration INT,
        starttime DATETIME,
        stoptime DATETIME,
        start_station_id INT,
        start_station_name TEXT,
        start_lat FLOAT,
        start_lng FLOAT,
        end_station_id INT,
        end_station_name TEXT,
        end_lat FLOAT,
        end_lng FLOAT,
        usertype TEXT
    )""")
    main(conn, columns, filepath)
