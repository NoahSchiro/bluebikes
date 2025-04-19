import os
import sqlite3

import pandas as pd
import torch
from torch.utils.data import Dataset

from .stations import station_names, station_name_preprocess

class StationData(Dataset):
    def __init__(
        self,
        station_name,
        sql_path,
        train,
        time_window_hours=24,
        pred_horizon=1
    ):
        
        self.dtype = {
            "station": "string",       # Read station as string
            "departures": "int64",     # Read departures as int
            "arrivals": "int64"        # Read arrivals as int
        }

        if station_name not in station_names:
            print("Not a valid station!")
            exit(0)
        else:
            self.station_name = station_name
            self.station_name_fp = station_name_preprocess(station_name)

        self.db_path = sql_path
        # CSV can go where the DB is but should be annotated with
        # station name whether it is train or test
        self.csv_path = os.path.join(
            os.path.dirname(sql_path),
            self.station_name_fp + ("_train.csv" if train else "_test.csv")
        )

        self.time_window_hours = time_window_hours
        self.pred_horizon = pred_horizon

        if os.path.exists(self.csv_path):
            self._load_csv(self.csv_path)
        elif os.path.exists(self.db_path):
            self._create_csv_from_sql(self.csv_path, self.db_path, train)
        else:
            print(f"Couldn't find a csv at {self.csv_path}, and no valid sql path was provided... failing")
            exit(0)
        
        # Extract all hours
        self.unique_hours = list(self.grouped_data["hour"].unique())
        self.unique_hours.sort()

    def _load_csv(self, path):
        self.data = pd.read_csv(
            path,
            dtype=self.dtype,
        )
        self.grouped_data = self.data.groupby("hour")
    
    def _create_csv_from_sql(self, csv_path, sql_path, train):
        db = sqlite3.connect(sql_path)

        where_query = ""

        if train:
            where_query = "hour < date('2024-01-01')"
        else:
            where_query = "hour >= date('2024-01-01')"

        print(f"Querying db at {sql_path}...")

        df = pd.read_sql_query(rf"""
            WITH Departures AS (
                SELECT
                    strftime('%Y-%m-%d %H:00:00', starttime) AS hour,
                    start_station_name AS station,
                    COUNT(*) AS departures
                FROM
                    blue_bikes
                WHERE {where_query}
                GROUP BY
                    hour, station
            ),
            Arrivals AS (
                SELECT
                    strftime('%Y-%m-%d %H:00:00', starttime) AS hour,
                    end_station_name AS station,
                    COUNT(*) AS arrivals
                FROM
                    blue_bikes
                WHERE {where_query}
                GROUP BY
                    hour, station
            )
            SELECT
                d.hour,
                d.station,
                d.departures,
                a.arrivals
            FROM
                Departures d
            JOIN
                Arrivals a ON d.hour = a.hour AND d.station = a.station
            WHERE d.station = '{self.station_name.replace("'", "''")}'
            ORDER BY
                d.hour;
        """, db, dtype=self.dtype)

        df = df.drop(["station"], axis=1)
        data         = df.sort_values(["hour"])
        grouped_data = data.groupby("hour")

        # Make sure everything is a datetime object
        data["hour"] = pd.to_datetime(data["hour"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

        # Remove N/A columns and replace with 0s
        data = data.dropna()
        data.set_index('hour', inplace=True)
        data = data.resample('H').asfreq().fillna(0)
        data.reset_index(inplace=True)

        # Extract datetime features
        data['year'] = data['hour'].dt.year
        data['month'] = data['hour'].dt.month
        data['day'] = data['hour'].dt.day
        data['weekday'] = data['hour'].dt.dayofweek
        data['hour_int'] = data['hour'].dt.hour

        print(f"Done. Saving to {csv_path}...")
        data.to_csv(csv_path, index=False)
        self._load_csv(csv_path)

    def __len__(self):
        return max(len(self.unique_hours) - self.time_window_hours - self.pred_horizon, 0)

    def __getitem__(self, idx):

        if idx < 0 or idx >= len(self.grouped_data) - self.time_window_hours - self.pred_horizon:
            raise IndexError(f"Index {idx} is out of range for the dataset.")

        # Get the hours corresponding to the input index and the input index plus the time window size
        x = self.data[idx:idx+self.time_window_hours]
        y = self.data[idx+self.time_window_hours:idx+self.time_window_hours+self.pred_horizon]
        
        # Convert to tensors
        tensor_x = torch.tensor(
            x[['year', 'month', 'day', 'weekday', 'hour_int', 'departures', 'arrivals']].values,
            dtype=torch.float32
        )
        tensor_y = torch.tensor(
            y[['year', 'month', 'day', 'weekday', 'hour_int', 'departures', 'arrivals']].values,
            dtype=torch.float32
        )

        # Will have shape [time_window, features]
        # and [time_window_features]
        return tensor_x, tensor_y 

if __name__=="__main__":

    # Pathing assumes you are running this from bluebikes/src
    data = StationData(
        station_name="Packard's Corner - Commonwealth Ave at Brighton Ave",
        sql_path="../data/data.db",
        train=False
    )

    x, y = data[0]
    print(int(y[0][6].item()))
