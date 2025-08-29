import os

# Get the current working directory of the script
current_dir = os.getcwd()

# Construct the full path to the __init__.py file
init_file_path = os.path.join(current_dir, "alphaDataBank", "utils", "__init__.py")

# Create the __init__.py file
with open(init_file_path, "w") as f:
    f.write("")
