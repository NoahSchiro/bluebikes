import sqlite3

loc = input("Db location: ")

connection = sqlite3.connect(loc)
cursor = connection.cursor()

while True:

    user_input = input("> ")

    try:
        res = cursor.execute(user_input).fetchall()
        for r in res:
            print(r)
    except Exception as e:
        print(e)


