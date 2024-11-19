import json
import os

BASE_DIR = os.path.dirname(__file__)


def add_message(new_item, file_name):
    try:
        file_path = os.path.join(BASE_DIR, file_name)

        # Check if the file exists
        if not os.path.exists(file_path):
            # Ensure each item has an ID when creating a new file
            new_item["Id"] = 1
            data = [new_item]

            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)

        else:
            # Load existing data from the file
            with open(file_path, "r") as file:
                data = json.load(file)

            # Assign a unique ID to the new item
            if data:
                # If the data already contains IDs, increment the highest ID
                max_id = max(item.get("Id", 0) for item in data)
                new_item["Id"] = max_id + 1
            else:
                # Start from 1 if the file is empty
                new_item["Id"] = 1

            # Append the new item to the list
            data.append(new_item)

            # Save the updated data back to the file
            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)

    except FileNotFoundError:
        print(
            f"File {file_name} not found. Make sure it exists before adding messages."
        )
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_name}: {e}")
        raise
    except (IOError, OSError) as e:
        print(f"Error accessing or writing to file {file_name}: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def add_messages(new_items, file_name):
    try:
        file_path = os.path.join(BASE_DIR, file_name)

        # Check if the file exists
        if not os.path.exists(file_path):
            # Ensure each item has an ID when creating a new file
            for i, item in enumerate(new_items, 1):
                item["Id"] = i

            with open(file_path, "w") as file:
                json.dump(new_items, file, indent=4)
        else:
            # Load existing data from the file
            with open(file_path, "r") as file:
                data = json.load(file)

            # Determine the next ID
            if data:
                max_id = max(item.get("Id", 0) for item in data)
            else:
                max_id = 1

            # Assign unique IDs to each new item
            for new_item in new_items:
                max_id += 1
                new_item["Id"] = max_id

            # Append the new items to the data
            data.extend(new_items)

            # Save the updated data back to the file
            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)

    except FileNotFoundError:
        print(
            f"File {file_path} not found. Make sure it exists before adding messages."
        )
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
        raise
    except (IOError, OSError) as e:
        print(f"Error accessing or writing to file {file_path}: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
