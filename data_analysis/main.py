import argparse
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def total(cursor):
    """Count the number of entries in the database."""

    res = cursor.execute("""
        SELECT count(*) from 'blue_bikes';
    """).fetchall()

    print(f"Total number of entries: {res[0][0]}")

def gender(cursor):
    """How many entries have the gender field."""

    res = cursor.execute("""
        SELECT gender AS g, count(gender) from 'blue_bikes' GROUP BY gender;
    """).fetchall()

    print(f"Male ridership: {res[2][1]}")
    print(f"Female ridership: {res[3][1]}")
    print(f"{res[1][1]} declined to provide this data")

def birth_year(db):
    """Distribution of rider age."""
    df = pd.read_sql_query("""
        SELECT
            strftime('%Y', starttime) - birth_year AS age,
            COUNT(*) as age_count
        FROM
            'blue_bikes'
        WHERE
            birth_year GLOB '[0-9]*' AND
            age < 100
        GROUP BY
            age
        ORDER by
            age;
    """, db)
    
    plt.figure(figsize=(14, 7), facecolor="#fafafa")
    ax = plt.gca()
    ax.set_facecolor("#fafafa")
    plt.bar(df['age'], df['age_count'], color='skyblue')

    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Rider count', fontsize=12)
    plt.title('Distribution of rider age', fontsize=14)
    plt.xticks(range(10, 100, 5))

    # Show the plot
    plt.tight_layout()
    plt.show()


def ridership_month(db):
    """Plot the ridership each month."""
    df = pd.read_sql_query("""
        SELECT 
            strftime('%Y-%m', starttime) AS month,
            COUNT(*) AS entry_count
        FROM 
            blue_bikes
        GROUP BY 
            strftime('%Y-%m', starttime)
        ORDER BY 
            month;
    """, db)

    plt.figure(figsize=(14, 7), facecolor="#fafafa")
    ax = plt.gca()
    ax.set_facecolor("#fafafa")
    plt.bar(df['month'], df['entry_count'], color='skyblue')

    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Rides', fontsize=12)
    plt.title('Rides by month', fontsize=14)
    plt.xticks(rotation=90, ha='right')

    # Show the plot
    plt.tight_layout()
    plt.show()

def ridership_hour(db):
    """Plot the ridership by hour of the day."""
    df = pd.read_sql_query("""
        SELECT 
            strftime('%H', starttime) AS hour,
            COUNT(*) AS entry_count
        FROM 
            blue_bikes
        GROUP BY 
            strftime('%H', starttime)
        ORDER BY 
            hour;
    """, db)

    plt.figure(figsize=(14, 7), facecolor="#fafafa")
    ax = plt.gca()
    ax.set_facecolor("#fafafa")
    plt.bar(df['hour'], df['entry_count'], color='skyblue')
    
    plt.xlabel('Hour', fontsize=12)
    plt.ylabel('Number of Rides', fontsize=12)
    plt.title('Ridership by Hour', fontsize=14)
 
    # Show the plot
    plt.tight_layout()
    plt.show()

def ridership_hour_gender(db):
    """Plot the ridership by hour of the day and gender (if reported)."""
    df = pd.read_sql_query("""
        SELECT
            gender,
            strftime('%H', starttime) AS hour,
            COUNT(*) AS entry_count
        FROM 
            blue_bikes
        WHERE
            gender = 1 OR gender = 2
        GROUP BY
            gender,
            strftime('%H', starttime)
       ORDER BY 
            hour;
    """, db)
    pivot_table = df.pivot(index='hour', columns='gender', values='entry_count').fillna(0)
 
    # Rename columns for clarity
    pivot_table.columns = ['Male', 'Female']

    pivot_table['Ratio'] = pivot_table['Male'] / pivot_table['Female'].replace(0, float('nan'))
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot ridership counts (Male and Female) on the primary y-axis (ax1)
    pivot_table[['Male', 'Female']].plot(
        kind='bar',
        ax=ax1,
        width=0.8,
        position=1,
        colormap='coolwarm',
        edgecolor='black'
    )
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('Ridership Count', fontsize=12, color='black')
    ax1.set_title('Male and Female Ridership with Male-to-Female Ratio by Hour', fontsize=14)
    ax1.set_xticklabels(pivot_table.index, rotation=0)

    # Create a secondary y-axis (ax2) to plot the Ratio
    ax2 = ax1.twinx()
    ax2.plot(
        pivot_table.index,
        pivot_table['Ratio'],
        marker='o',
        linestyle='-',
        color='purple',
        label='Male-to-Female Ratio'
    )
    ax2.set_ylabel('Male-to-Female Ratio', fontsize=12, color='purple')
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

def station_popularity(cursor):
    """Order stations from most popular to least popular."""
    res = cursor.execute("""
        SELECT 
            start_station_name AS start_loc,
            COUNT(*) AS count
        FROM 
            blue_bikes
        GROUP BY 
            start_station_name
        ORDER BY 
            count DESC
        LIMIT 20;
    """).fetchall()
    
    print("Most popular stations to start from: ")
    for r in res:
        print(f"{r[1]:<} {r[0]:<}")

    res = cursor.execute("""
        SELECT 
            end_station_name AS end_loc,
            COUNT(*) AS count
        FROM 
            blue_bikes
        GROUP BY 
            end_station_name
        ORDER BY 
            count DESC
        LIMIT 20;
    """).fetchall()
    
    print("Most popular stations to end at: ")
    for r in res:
        print(f"{r[1]:<} {r[0]:<}")

def route_popularity(cursor):
    """Determine what routes are most popular."""
    res = cursor.execute("""
        SELECT 
            start_station_name, 
            end_station_name, 
            COUNT(*) AS count
        FROM 
            blue_bikes
        GROUP BY 
            end_station_name, 
            start_station_name
        ORDER BY 
            count DESC
        LIMIT 20;
    """).fetchall()

    for r in res:
        print(f"{r[2]:<} {r[0]} -> {r[1]}")

def trip_duration(db):
    """Distribution of trip duration."""
    df = pd.read_sql_query("""
        SELECT 
            CAST(tripduration / 60 as INTEGER) AS minute_buckets,
            COUNT(*) AS count
        FROM
            blue_bikes 
        WHERE
            minute_buckets < 100 AND minute_buckets > 1
        GROUP BY
            minute_buckets
        ORDER BY
            minute_buckets;
    """, db)
    
    plt.figure(figsize=(14, 7), facecolor="#fafafa")
    ax = plt.gca()
    ax.set_facecolor("#fafafa")
    plt.bar(df['minute_buckets'], df['count'], color='skyblue')

    plt.xlabel('Duration', fontsize=12)
    plt.ylabel('Rider count', fontsize=12)
    plt.title('Distribution of ride duration', fontsize=14)
    plt.xticks(range(0, 100, 5))

    # Show the plot
    plt.tight_layout()
    plt.show()

def trip_length(db):
    """Distribution of trip length."""
    print("TODO")

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Analyze bluebike data."
    )

    parser.add_argument(
        "--db", help="Location of the data base",
        default="/home/noah/Work/bluebikes/data/data.db"
    )

    parser.add_argument(
        "--total", help="Total number of entries",
        action="store_true"
    )
    parser.add_argument(
        "--gender", help="Total number of entries with gender field",
        action="store_true"
    )
    parser.add_argument(
        "--birth_year", help="Total number of entries with birth year field",
        action="store_true"
    )
    parser.add_argument(
        "--ridership_month", help="Plot ridership by each month",
        action="store_true"
    )
    parser.add_argument(
        "--ridership_hour", help="Plot ridership by each hour of the day",
        action="store_true"
    )
    parser.add_argument(
        "--ridership_hour_gender", help="Plot ridership by each hour of the day and by gender if reported",
        action="store_true"
    )
    parser.add_argument(
        "--station_popularity", help="Find what stations are most popular for leaving / arriving",
        action="store_true"
    )
    parser.add_argument(
        "--route_popularity", help="Find what routes are most popular",
        action="store_true"
    )
    parser.add_argument(
        "--trip_duration", help="Show distribution of trip duration",
        action="store_true"
    )
    parser.add_argument(
        "--trip_length", help="Show distribution of trip lengths",
        action="store_true"
    )

    args = parser.parse_args()

    db = sqlite3.connect(args.db)
    cursor = db.cursor()

    if args.total:
        total(cursor)
    if args.gender:
        gender(cursor)
    if args.birth_year:
        birth_year(db)
    if args.ridership_month:
        ridership_month(db)
    if args.ridership_hour:
        ridership_hour(db)
    if args.ridership_hour_gender:
        ridership_hour_gender(db)
    if args.station_popularity:
        station_popularity(cursor)
    if args.route_popularity:
        route_popularity(cursor)
    if args.trip_duration:
        trip_duration(db)
    if args.trip_length:
        trip_length(db)
