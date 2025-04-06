import os
import sqlite3

import pandas as pd
import numpy as np
import torch

class BikeData():
    def __init__(
        self,
        csv_path,
        sql_path,
        train,
        time_window_hours=24,
        pred_horizon=1,
        batch_size=4
    ):
        
        self.dtype = {
            "station": "string",       # Read station as string
            "departures": "int64",     # Read departures as int
            "arrivals": "int64"        # Read arrivals as int
        }

        self.csv_path = csv_path
        self.db_path = sql_path

        self.time_window_hours = time_window_hours
        self.pred_horizon = pred_horizon
        self.batch_size = batch_size 

        if os.path.exists(self.csv_path):
            self._load_csv(csv_path)
        elif os.path.exists(self.db_path):
            self._create_csv_from_sql(self.csv_path, self.db_path, train)
        else:
            print(f"Couldn't find a csv at {self.csv_path}, and no valid sql path was provided... failing")
            exit(0)

        # Make sure everything is a datetime object
        self.data["hour"] = pd.to_datetime(self.data["hour"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

        # Remove N/A columns and replace with 0s
        self.data = self.data.dropna()

        # Extract all station names, create embeddings
        self.station_names = list(self.data["station"].unique())
        self.station_names.sort()
        self._create_station_embeddings(self.station_names)

        # Extract all hours
        self.unique_hours = list(self.grouped_data["hour"].unique())

        # Now we need to fill in values which might have no entries.
        # If a particular station at a particular time has no arrivals
        # or entries, then it won't appear in this list. We want it
        # to appear, and just have a 0
        full_range = pd.date_range(start=self.data["hour"].min(), end=self.data["hour"].max(), freq="H")

        # Create a MultiIndex with all (station, timestamp) combinations
        multi_index = pd.MultiIndex.from_product(
            [full_range, self.station_names],
            names=["hour", "station"]
        )

        # Reindex DataFrame to ensure all combinations exist
        self.data = self.data.set_index(["hour", "station"]).reindex(multi_index, fill_value=0).reset_index()

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

        df = pd.read_sql_query(f"""
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
            ORDER BY
                d.hour, d.station;
        """, db, dtype=self.dtype)

        self.data         = df.sort_values(["hour"])
        self.grouped_data = self.data.groupby("hour")

        print(f"Done. Saving to {csv_path}...")
        df.to_csv(csv_path, index=False)
        print("Done.")

    def _create_station_embeddings(self, stations):
        self.station_embeddings = {station: torch.rand(10) for station in stations}

    def __len__(self):
        return len(self.grouped_data) - self.time_window_hours - self.pred_horizon

    def __getitem__(self, idx):
        """
        For this particular hour, we randomly select `batch_size` stations and the data for that station.
        We want the data from `time_window_hours + pred_horizon` into the future as well.
        """

        if idx < 0 or idx >= len(self.grouped_data) - self.time_window_hours - self.pred_horizon:
            raise IndexError(f"Index {idx} is out of range for the dataset.")

        # Get the hours corresponding to the input index and the input index plus the time window size
        start_idx = self.unique_hours[idx][0]
        end_idx   = self.unique_hours[idx + self.time_window_hours+self.pred_horizon][0]

        time_mask = (self.data["hour"] >= start_idx) & (self.data["hour"] <= end_idx)
        hour_group = self.data[time_mask]
        
        # Randomly select stations
        station_names = list(hour_group["station"].unique())
        sampled_stations = np.random.choice(station_names, size=self.batch_size, replace=False)
        
        # Filter to only stations in sampled stations
        filtered = hour_group[hour_group["station"].isin(sampled_stations)]

        all_station_data = [
            filtered[filtered["station"] == station]
            for station in sampled_stations
        ]

        times = pd.to_datetime(hour_group["hour"].unique())
        years = list(times.year)
        months = list(times.month)
        days = list(times.day)
        hour = list(times.hour)

        time_info = list(zip(years, months, days, hour))

        # Converts times into a tensor, and copys the data across many batches
        # [batch_size, sequence_length, time_information]
        times_as_tensor = torch.tensor(time_info).unsqueeze(0).expand(self.batch_size, -1, -1)

        # Get station information in embedding form
        # torch.shape[batch_size, sequence_length, embedding_dim=10]
        stations = torch.stack([
            torch.stack([self.station_embeddings[x] for x in s["station"].to_list()])
            for s in all_station_data
        ])

        # torch.shape[batch_size, sequence_length, arrivals = 1]
        arrivals = torch.stack(
            [torch.tensor(s["arrivals"].to_list()) for s in all_station_data]
        ).unsqueeze(-1)

        # torch.shape[batch_size, sequence_length, depatures=1]
        departures = torch.stack(
            [torch.tensor(s["departures"].to_list()) for s in all_station_data]
        ).unsqueeze(-1)
        
        # Final shape
        # batch, time_seq_input, station_embedding, time information, arrivals, departures
        # batch, time_seq_output, station_embedding, time information, arrivals, departures
        all_as_tensor = torch.cat((stations, times_as_tensor, arrivals, departures), dim=2)
        return all_as_tensor



if __name__=="__main__":

    # Pathing assumes you are running this from bluebikes/src
    pd.set_option('display.max_columns', 7)

    data = BikeData(
        csv_path="../data/data_test.csv",
        sql_path="../data/data.db",
        train=False,
        batch_size=1
    )
    import datetime
    start = datetime.datetime.now()
    print(data[2000][0])
    print(data[2001][0])

