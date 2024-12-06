import os
import csv

def find_and_print_csv_first_rows(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                try:
                    with open(csv_path, mode='r', newline='', encoding='utf-8') as csv_file:
                        reader = csv.reader(csv_file)
                        first_row = next(reader, None)
                        if first_row:
                            print(f"{first_row} ({csv_path})")
                        else:
                            print(f"{csv_path} is empty.")
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")

if __name__ == "__main__":
    directory_to_search = input("Enter the directory to search for CSV files: ")
    if os.path.isdir(directory_to_search):
        find_and_print_csv_first_rows(directory_to_search)
    else:
        print("Invalid directory. Please enter a valid path.")
