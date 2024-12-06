import os
import zipfile

def extract_and_delete_zips(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    # Get a list of all ZIP files in the directory
    zip_files = [f for f in os.listdir(directory) if f.endswith('.zip')]

    if not zip_files:
        print("No ZIP files found in the directory.")
        return

    for zip_file in zip_files:
        zip_path = os.path.join(directory, zip_file)
        try:
            # Extract the ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                extract_path = os.path.join(directory, os.path.splitext(zip_file)[0])
                zip_ref.extractall(extract_path)
                print(f"Extracted: {zip_file} to {extract_path}")

            # Delete the ZIP file
            os.remove(zip_path)
            print(f"Deleted: {zip_file}")
        
        except zipfile.BadZipFile:
            print(f"Error: {zip_file} is not a valid ZIP file.")
        except Exception as e:
            print(f"An error occurred with {zip_file}: {e}")

if __name__ == "__main__":
    directory = input("Enter the path to the directory: ").strip()
    extract_and_delete_zips(directory)
