import sqlite3
from glob import glob

import torch

from src.stations import station_name_preprocess, station_names
from src.data import StationData
from src.model import BlueBikesModel

unlock_cost = 2.95
year_norm = {
    "year" : {
        "max" : 2025,
        "min" : 2015
    }
}
elasticity = -0.4

def dynamic_price_modify(supply, demand):
    if supply == 0:
        supply = 1.0
    if demand == 0:
        demand = 1.0
    
    return (supply / demand) ** (1 / -0.2)

# Count station with no dynamic pricing
def count_station(station_ds):
    departures = 0
    for _, y in station_ds:
        departures += int(y[0][6].item())
    
    print(f"Departures at station {station_ds.station_name}:")
    print(f"{departures} (revenue: ${unlock_cost * departures:,.2f})")

    return departures

# Count station with dynamic pricing
def count_station_dynamic(model, station_ds):

    # Accumulate number of departures and revenue for this station
    # Both with normal pricing and dynamic pricing
    departures_normal = 0
    departures_dynamic = 0
    normal_revenue = 0.
    dynamic_revenue = 0.
    
    for x, y in station_ds:
        x = x.unsqueeze(0) # Add dimension to input
        y = int(y[0][6].item()) # Extract only departures

        # Use the prediction of the next hour to set the price
        with torch.no_grad():
            arrival, departure = model.inference(x, year_norm)[0]
        
        price_mod = dynamic_price_modify(arrival, departure)
        demand_mod = price_mod ** elasticity

        new_price = unlock_cost * price_mod
        # With the new price, how does this effect the actual demand? (rounded)
        new_departures = int(y * demand_mod)

        departures_normal += y
        departures_dynamic += new_departures

        normal_revenue += (y * unlock_cost)
        dynamic_revenue += (new_departures * new_price)
    
    return departures_normal, departures_dynamic, normal_revenue, dynamic_revenue

def pricing_with_datastructure():

    total_departures_normal = 0
    total_departures_dynamic = 0

    total_revenue_normal = 0.
    total_revenue_dynamic = 0.

    for station in station_names:
        station_ds = StationData(
            station_name=station,
            sql_path="data/data.db",
            train=False # This will fetch data from 2024
        )

        # If no data for 2024
        if len(station_ds) == 0:
            continue

        # If we trained a model on this station
        station_clean = station_name_preprocess(station)
        models = glob(f"./models/{station_clean}*.pth")
        if len(models) == 0:
            print(f"No model for {station}")
            departures = count_station(station_ds)
            total_departures_normal += departures
            total_revenue_normal += departures * unlock_cost
        else:
            model = BlueBikesModel(
                input_size=7,
                hidden_size=16,
                num_layers=4
            )
            model.load_state_dict(torch.load(models[0], weights_only=True))
            print(f"Loaded model for {station}")
            
            dep_norm, dep_dyn, rev_norm, rev_dyn = count_station_dynamic(model, station_ds)
            
            print(f"Departures (normal, dynamic): {dep_norm:,}, {dep_dyn:,}")
            print(f"Revenue (normal, dynamic): ${rev_norm:,.2f}, ${rev_dyn:,.2f}")
            total_departures_normal += dep_norm
            total_departures_dynamic += dep_dyn
            total_revenue_normal += rev_norm
            total_revenue_dynamic += rev_dyn

    print("Normal:")
    print(f"Departures: {total_departures_normal:,}")
    print(f"Revenue: ${total_revenue_normal:,.2f}\n")
    print("Dynamic:")
    print(f"Departures: {total_departures_dynamic:,}")
    print(f"Revenue: ${total_revenue_dynamic:,.2f}")


if __name__=="__main__":

    db = sqlite3.connect("/home/noah/Work/bluebikes/data/data.db")
    cursor = db.cursor()

    start = "\'2024\'" #10:18:15
    pricing_with_datastructure()
