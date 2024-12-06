import sqlite3

connection = sqlite3.connect("../data/data.db")
cursor = connection.cursor()

while True:

    user_input = input("> ")

    res = cursor.execute(user_input).fetchall()

    for r in res:
        print(res)
