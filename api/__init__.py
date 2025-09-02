import os

# Get the directory of the current file (api/__init__.py)
api_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the project root
project_root = os.path.dirname(api_dir)

# Construct a reliable path to the utils/__init__.py file
init_file_path = os.path.join(project_root, "utils", "__init__.py")

# Ensure the directory exists and create an empty __init__.py file
# This ensures the 'utils' directory is treated as a Python package.
os.makedirs(os.path.dirname(init_file_path), exist_ok=True)
with open(init_file_path, "a") as f:
    pass # 'a' mode will create the file if it doesn't exist and do nothing if it does