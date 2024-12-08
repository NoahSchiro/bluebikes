# Bluebikes

The City of Boston has [BlueBikes](https://bluebikes.com/). Kinda like NYC's CitiBikes. The City also provides [data](https://bluebikes.com/system-data) of _every trip ever going back to 2011_! What kind of interesting stuff can we do with that?

# Get goin'

First you need to download the data [here](https://s3.amazonaws.com/hubway-data/index.html). Each month is broken into it's own zip file. Drop them into a directory (such as `./data/`). Once you have that downloaded, you can extract and delete the zips with:

```bash
python3 ./data_preprocess/unzip.py
```

Unfortunately, and inexplicably, some of the months have a slightly different csv layout. You can verify this for yourself by running `python3 ./data_preprocess/header.py`. We will need to clean this up and consolidate all of the data into one tidy and neat place. I chose to use a sqlite database.

```bash
python3 ./data_preprocess/consolidate.py
```

And out will pop a database with one table: `blue bikes`. If you want to explore this data by itself without doing any cool machine learning stuff on it, I built a tiny SQL terminal that you can use with:

```bash
python3 ./data_preprocess/db.py
```

This will give you an interactive terminal where you can run SQL commands.

# Dynamic Pricing

We will try to build a Dynamic Pricing Model to increase revenue and bike availability.

# Data Analysis

We have a special [doc](./data_analysis/README.md) for our analysis of the data!
