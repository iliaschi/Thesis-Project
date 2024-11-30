# Here we will generate the data for the thesis

### 1
# We will have a folder checker that will chech if the folder of the data exists
# If it does not exist, it will create the folder

### 2
# Then we will as for an input from the user of Yes and No

### 3
# Then the number of data to be generated


import os
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def validate_folder_name(folder_name):
    """
    Validate the folder name for invalid characters or empty input.
    """
    if not folder_name or any(char in folder_name for char in r'<>:"/\|?*'):
        print("Invalid folder name. Please try again.")
        return False
    return True

def list_folder_contents(folder_name):
    """
    List the contents of a folder if it exists.
    """
    contents = os.listdir(folder_name)
    if contents:
        print(f"Contents of '{folder_name}':")
        for item in contents:
            print(f"  - {item}")
    else:
        print(f"The folder '{folder_name}' is empty.")

def create_folder(folder_name):
    """
    Create a folder and log the action.
    """
    try:
        os.makedirs(folder_name)
        print(f"The folder '{folder_name}' has been created.")
        logging.info(f"Folder '{folder_name}' created successfully.")
    except PermissionError:
        print("Error: Insufficient permissions to create the folder.")
        logging.error(f"Failed to create folder '{folder_name}': Permission denied.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.error(f"Failed to create folder '{folder_name}': {e}")

def main():
    """
    Main function to manage folder creation and checking.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Folder management script.")
    parser.add_argument("--folder", type=str, help="Name of the folder.")
    parser.add_argument("--create", action="store_true", help="Create the folder if it doesn't exist.")
    args = parser.parse_args()

    # If no folder is provided via arguments, ask the user
    folder_name = args.folder
    if not folder_name:
        folder_name = input("Enter the name of the folder to save data (default: 'data'): ").strip() or "data"

    # Validate folder name
    if not validate_folder_name(folder_name):
        return

    # Check if folder exists
    if os.path.exists(folder_name):
        print(f"The folder '{folder_name}' already exists.")
        logging.info(f"Folder '{folder_name}' already exists.")
        list_folder_contents(folder_name)
    else:
        print(f"The folder '{folder_name}' does not exist.")
        if args.create:
            create_folder(folder_name)
        else:
            create = input("Do you want to create it? (yes/no): ").strip().lower()
            if create in ['yes', 'y']:
                create_folder(folder_name)
            else:
                print(f"The folder '{folder_name}' was not created. User declined.")
                logging.info(f"User declined to create folder '{folder_name}'.")


# Now we will create the part where we extract the data from the 
# Ai synthetic image data generator 



if __name__ == "__main__":
    main()
