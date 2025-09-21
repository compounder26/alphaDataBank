"""
Bootstrap utilities for AlphaDataBank project.

This module provides utilities for setting up the Python path consistently
across all project scripts, eliminating the need for repeated sys.path
manipulation in individual files.
"""
import sys
import os


def setup_project_path():
    """
    Standard path setup for all project scripts.

    This function automatically detects the project root directory by looking
    for marker files that indicate the project root and adds it to sys.path
    if not already present.

    Returns:
        str: The absolute path to the project root directory
    """
    # Start from the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir

    # Marker files that indicate project root (in order of preference)
    marker_files = [
        '.git',           # Git repository root (most reliable)
        'requirements.txt',  # Python project dependency file
        'setup.py',       # Python package setup file
        'README.md',      # Project documentation
        '.env.example',   # Environment template file
    ]

    # Walk up the directory tree until we find a marker file
    while True:
        # Check if any marker file exists at this level
        for marker in marker_files:
            if os.path.exists(os.path.join(project_root, marker)):
                # Found a marker - this is our project root
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                return project_root

        # Move up one directory
        parent_dir = os.path.dirname(project_root)
        if parent_dir == project_root:  # Reached filesystem root
            raise RuntimeError(f"Could not find project root (none of {marker_files} found)")
        project_root = parent_dir


def get_project_root():
    """
    Get the project root directory without modifying sys.path.

    Returns:
        str: The absolute path to the project root directory
    """
    # Start from the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir

    # Marker files that indicate project root (in order of preference)
    marker_files = [
        '.git',           # Git repository root (most reliable)
        'requirements.txt',  # Python project dependency file
        'setup.py',       # Python package setup file
        'README.md',      # Project documentation
        '.env.example',   # Environment template file
    ]

    # Walk up the directory tree until we find a marker file
    while True:
        # Check if any marker file exists at this level
        for marker in marker_files:
            if os.path.exists(os.path.join(project_root, marker)):
                # Found a marker - this is our project root
                return project_root

        # Move up one directory
        parent_dir = os.path.dirname(project_root)
        if parent_dir == project_root:  # Reached filesystem root
            raise RuntimeError(f"Could not find project root (none of {marker_files} found)")
        project_root = parent_dir