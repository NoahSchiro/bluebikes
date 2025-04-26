# Bluebikes
The City of Boston has [BlueBikes](https://bluebikes.com/). Similar to NYC's CitiBikes. The City also provides [data](https://bluebikes.com/system-data) of _every trip ever going back to 2011_! What kind of interesting stuff can we do with that?

## Get started

First you need to download the data [here](https://s3.amazonaws.com/hubway-data/index.html). Each month is broken into it's own zip file. Drop them into a directory (such as `./data/`). Once you have that downloaded, you can extract and delete the zips with:

```bash
python3 ./data_preprocess/unzip.py
```

Unfortunately, and inexplicably, some of the months have a slightly different CSV layout. You can verify this for yourself by running `python3 ./data_preprocess/header.py`. We will need to clean this up and consolidate all of the data into one tidy and neat place. I chose to use a sqlite database.

```bash
python3 ./data_preprocess/consolidate.py
```

And out will pop a database with one table: `blue_bikes`. If you want to explore this data by itself without doing any cool machine learning stuff on it, I built a tiny SQL terminal that you can use with:

```bash
python3 ./data_preprocess/db.py
```

This will give you an interactive terminal where you can run SQL commands.

## Training

Training the model is relatively straightforward. I am training a model for each individual station. Each model is an LSTM with a linear prediction head attached. I am using a 24 hour context window and only predicting the next hour. The primary metric that I want to optimize for is absolute error in the amount of arrivals / departures that a given station has.

I trained a model for roughly 212 stations. Here is a breakdown of how many models fell into each range of error:

| Error | Departure errors in this range | Arrival errors in this range |
|-------|--------------------------------|------------------------------|
| `e > 2.0` | 14 | 12 |
| `2.0>e>1.0` | 128 | 130 |
| `1.0 > e` | 70 | 70 |

For models exceeding 2.0, we tend to have a very large dataset. I believe that increasing the model size (either hidden size or number of layers in the LSTM) would improve this model, but I would also have to change the way that I am saving / loading the model such that I can have variable model architectures. This is an area for future exploration.

### Models

I have trained for the most popular stations (anything > 15k arrivals / departures in 2024). If you would like to use these for your own applications, they can be found [here](https://drive.google.com/drive/folders/1xTUIVHAwcvt1qPb1tTZTNizWnu7aTDcw?usp=sharing).

The models must have the following architecture:

```
# BlueBikesModel is defined in src/model.py
model = BlueBikesModel(
    input_size=7,
    hidden_size=16,
    num_layers=4
)
```

Input is a tensor of shape: `[batch, sequence_length, features]`. The features are `[year, month, day, day of week, hour, arrivals, departures]`. I have trained on a sequence length of 24. The model will output the prediction for the subsequent hour with shape: `[arrivals, departures]` (batched). Some of the date normalization happens automatically, but years are normalized within a defined range which is the range you would like the model to be operational. For these, I have used the minimum year to be 2015 and the maximum year to be 2025. Somewhere you will need the following defined as the model will use it in preprocessing:

```
year_norm = {
    "year" : {
        "max" : 2025,
        "min" : 2015
    }
}
```

## Dynamic Pricing

Setting dynamic pricing can be divided into two steps:

1. Predicting future supply and demand (this is what the ML model does).
2. Using economic models to predict how people will act in response to a price point.

We will use a very simple economic model for the second step. We will assume that the demand moves linearly with respect to price (in reality this is almost never the case) and that the slope (oftentimes just called the price elasticity of demand) of this line is -0.4. I am using -0.4 from Uber. They found the price elasticity of demand for their service was roughly -0.4. Since a ride-share bike service is roughly the same thing, we will use the same number here to simulate how the users behave with respect to the new price.

Our model will predict the supply and demand (arrival and departure of bikes) in the subsequent hour and try to set a price such that the number of arrivals and departures is roughly the same. This has a few benefits:

1. It will help keep bikes evenly distributed through the network. This will reduce the number of stations completely full and completely empty and will also reduce the amount of times the City has to transport bikes manually to redistribute. This saves on costs.
2. It will ensure greater availability of bikes to users.
3. It will increase revenue to this city and to the program, which can be used to expand public transportation services such as the BlueBikes program.

I simulated the revenue for 2024 with normal pricing and dynamic pricing on the 212 stations I trained for.
|            | Normal pricing | Dynamic Pricing |
|------------|----------------|-----------------|
| Departures | 3,827,649      | 3,632,203       |
| Revenue    | $11,291,564.55 | $17,908,166.06  |

Total usage of the network only fell by about 5%, but revenue jumped 58% or $6.6 million

## Data Analysis

We have a special [doc](./data_analysis/README.md) for our analysis of the data!
