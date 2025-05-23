import argparse
import os
from datetime import datetime
from glob import glob

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm
import mlflow

from src.model import BlueBikesModel
from src.data import StationData
from src.stations import station_names, stations_of_consequence, station_name_preprocess

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def train(model, loss_fn, optim, train_dl, data_stats, epoch):

    loss_acc = 0

    for (x, y) in tqdm(train_dl):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        x = model.preprocess(x, data_stats)
        y = model.preprocess(y, data_stats)
        y = torch.squeeze(y[:, :, -2:], dim=1)

        pred = model(x)

        loss = loss_fn(y, pred)
        loss_acc += loss.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

    train_loss = loss_acc / len(train_dl.dataset)
    mlflow.log_metric("train_loss", train_loss, step=epoch)

    print(f"Train loss: {train_loss}")
        
def evaluate(model, loss_fn, test_dl, test_stats, epoch):

    loss_acc = 0
    arrival_error_acc = 0
    departure_error_acc = 0
    
    with torch.no_grad():
        for (x, y) in tqdm(test_dl):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            x = model.preprocess(x, test_stats)
            y = model.preprocess(y, test_stats)
            y = torch.squeeze(y[:, :, -2:], dim=1)

            pred = model(x)

            abs_error = (y - pred.round()).abs()
            arrival_abs_error = abs_error[:, 0].sum().item()
            departure_abs_error = abs_error[:, 1].sum().item()

            arrival_error_acc += arrival_abs_error
            departure_error_acc += departure_abs_error
            
            loss = loss_fn(y, pred)
            loss_acc += loss.item()

    eval_loss = loss_acc / len(test_dl.dataset)
    arrival_error = arrival_error_acc / len(test_dl.dataset)
    departure_error = departure_error_acc / len(test_dl.dataset)
    mlflow.log_metric("eval_loss", eval_loss, step=epoch)
    mlflow.log_metric("arrival_error", arrival_error, step=epoch)
    mlflow.log_metric("departure_error", departure_error, step=epoch)

    print(f"Eval loss: {eval_loss}")
    print(f"Arrival average error: {arrival_error}")
    print(f"Departure average error: {departure_error}")
    return eval_loss, arrival_error, departure_error
 
def main(args):

    torch.manual_seed(args.seed)

    model = BlueBikesModel(
        input_size=7, # Fixed based on the BlueBikes dataset
        hidden_size=args.hidden_size,
        num_layers=args.layers
    ).to(DEVICE)

    loss_fn = nn.MSELoss()
    optim = Adam(model.parameters(), args.lr)

    train_ds = StationData(
        station_name=args.station,
        sql_path="./data/data.db",
        train=True,
        time_window_hours=args.time_window,
    )
    test_ds = StationData(
        station_name=args.station,
        sql_path="./data/data.db",
        train=False,
        time_window_hours=args.time_window,
    )

    if len(train_ds) == 0 or len(test_ds) == 0:
        print("Not enough training data, returning")
        return

    year_norm = {
        "year": {
            "min" : 2010,
            "max" : 2025
        },
    }

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=os.cpu_count()
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=os.cpu_count()
    )

    if not os.path.exists("./models"):
        os.mkdir("./models")

    # Convert the station name to something appropriate for linux file paths
    station_name = station_name_preprocess(args.station)
    time = datetime.now().strftime("%Y-%m-%d:%H-%M")
    save_path = f"./models/{station_name}{time}.pth"

    with mlflow.start_run():
        mlflow.log_params(vars(args))

        print("Intial performance: ")
        best_eval_loss, best_arr_err, best_dep_err = evaluate(model, loss_fn, test_dl, year_norm, 0)
        best_epoch = 0
        epoch = 0
        while True:
            print(f"EPOCH: {epoch}")
            train(model, loss_fn, optim, train_dl, year_norm, epoch)
            el, ae, de = evaluate(model, loss_fn, test_dl, year_norm, epoch)
            best_eval_loss = min(best_eval_loss, el)

            if ae + de < best_arr_err + best_dep_err:
                best_arr_err = ae
                best_dep_err = de
                best_epoch = epoch
                print("Best model, saving...")
                torch.save(model.state_dict(), save_path)

            # Early stopping if we haven't reached a better model in 5 epochs
            if epoch >= best_epoch + 5 or epoch >= args.max_epochs:
                mlflow.log_param("epochs", best_epoch)
                break

            epoch += 1

        mlflow.log_metric("best_eval_loss", best_eval_loss)
        mlflow.log_metric("best_arrival_error", best_arr_err)
        mlflow.log_metric("best_departure_error", best_dep_err)

# Logic for making a model for a list of stations
# In this case, I am only making a model for stations with more than
# 15k trips in 2024 (the test year). There is ~200 such stations.
def run_many_stations(args):
    for station in stations_of_consequence:

        # If I have already trained a station, skip it
        if glob(f"./models/{station_name_preprocess(station)}*.pth"):
            print(f"Skipping {station}")
        else:
            print(f"Training for station: {station}")
            args.station = station
            main(args)


if __name__=="__main__":

    mlflow.set_tracking_uri(uri="http://localhost:8080")
    mlflow.set_experiment("BlueBikes")

    parser = argparse.ArgumentParser(
        prog="BlueBikes demand forecasting training"
    )

    # Training parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed"
    )
    parser.add_argument(
        "-e", "--max_epochs",
        type=int,
        default=50,
        help="Maximum number of epochs"
    )
    parser.add_argument(
        "-b", "--batch",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )

    # Dataset options
    parser.add_argument(
        "--station",
        choices=list(station_names),
        help="Which station to train on"
    )
    parser.add_argument(
        "--time_window",
        type=int,
        default=24,
        help="The amount of context the model gets before predicting the next hour"
    )

    # Model hyperparameters
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=16,
        help="Hidden size of LSTM"
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help="Number of hidden layers in LSTM"
    )

    args = parser.parse_args()

    #main(args)
    run_many_stations(args)
