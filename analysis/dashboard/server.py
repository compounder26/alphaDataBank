"""
Dashboard Server Entry Point

Server startup and CLI handling for the Alpha Analysis Dashboard.
Extracted from visualization_server.py main() function with preserved functionality.
"""

import os
import sys
import argparse
import webbrowser
import threading
import time

# Add the project root to the path for imports (preserve original behavior)
# Setup project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.bootstrap import setup_project_path
setup_project_path()

from database.schema import initialize_analysis_database

from .app import create_visualization_app, set_operators_file, set_dynamic_operators_list
from .services import (
    load_clustering_data, load_operators_data, load_tier_specific_datafields,
    set_tier_operators_and_datafields
)
from .config import DASHBOARD_SETTINGS, DEFAULT_OPERATORS_FILE


def open_browser(port, delay=1):
    """
    Open browser after a delay.

    EXACT COPY from visualization_server.py lines 5181-5184
    """
    time.sleep(delay)
    webbrowser.open(f'http://localhost:{port}')


def main():
    """
    Run the analysis and visualization server.

    EXACT COPY from visualization_server.py main() function lines 5219-5313
    Preserves all CLI arguments and startup logic.
    """
    parser = argparse.ArgumentParser(description="Alpha Analysis & Clustering Dashboard")
    parser.add_argument("--port", type=int, default=DASHBOARD_SETTINGS['default_port'], help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--operators-file", type=str, help="Path to custom operators file (JSON or TXT)")

    args = parser.parse_args()

    # Set global file paths based on arguments (preserve original logic)
    if args.operators_file:
        set_operators_file(args.operators_file)
        print(f"Using dynamic operators file: {args.operators_file}")
    else:
        set_operators_file(DEFAULT_OPERATORS_FILE)
        print(f"Using default operators file: {DEFAULT_OPERATORS_FILE}")

    # Load dynamic operators and datafields if using JSON file (tier-specific from API)
    operators_file = args.operators_file or DEFAULT_OPERATORS_FILE
    if operators_file.endswith('.json'):
        try:
            # Load tier-specific operators from JSON
            operators_list = load_operators_data(operators_file)
            print(f"✅ Loaded {len(operators_list)} tier-specific operators from JSON file")

            # Load tier-specific datafields from database (populated by renew_genius.py)
            datafields_list = load_tier_specific_datafields()
            print(f"✅ Loaded {len(datafields_list)} tier-specific datafields from database")

            # Set both operators and datafields for tier-based filtering
            set_tier_operators_and_datafields(operators_list, datafields_list)
            print("✅ Tier-specific filtering enabled for alpha exclusion")

        except Exception as e:
            print(f"⚠️ Error loading tier-specific data: {e}")
            print("⚠️ WARNING: Running without tier filtering - ALL alphas will be shown!")
    else:
        print("⚠️ Not using dynamic operators file - tier filtering disabled")
        print("⚠️ Run with --renew or use renew_genius.py to enable tier-based filtering")

    # Always try to load clustering data automatically
    data = None
    # Auto-detect clustering data from standard locations
    print("Looking for clustering data files...")
    # The clustering data will be loaded by the dashboard components as needed

    # Create app using refactored factory
    print("Initializing dashboard...")
    app = create_visualization_app(data, getattr(sys.modules[__name__], 'DYNAMIC_OPERATORS_LIST', None))

    # Always start browser opener thread
    threading.Thread(target=open_browser, args=(args.port, DASHBOARD_SETTINGS['browser_delay_seconds']), daemon=True).start()

    # Run server (preserve original logic)
    mode = "Analysis & Clustering" if data else "Analysis-only"
    print(f"Starting {mode} dashboard on port {args.port}...")
    print(f"Dashboard URL: http://localhost:{args.port}")

    print(f"Browser will open automatically in {DASHBOARD_SETTINGS['browser_delay_seconds']} seconds...")

    try:
        app.run(debug=args.debug, port=args.port, host='127.0.0.1')
    except Exception as e:
        print(f"Error starting server: {e}")
        print("\nTroubleshooting tips:")
        print(f"- Check if port {args.port} is already in use")
        print("- Try a different port with --port argument")
        print("- Ensure all required dependencies are installed")


def create_production_app():
    """
    Create app for production deployment (WSGI).

    Returns:
        Dash app configured for production
    """
    return create_visualization_app()


def start_development_server(port: int = None, debug: bool = True, open_browser: bool = True):
    """
    Start development server with common settings.

    Args:
        port: Server port
        debug: Debug mode
        open_browser: Whether to open browser
    """
    if port is None:
        port = DASHBOARD_SETTINGS['default_port']

    app = create_visualization_app()

    if open_browser:
        threading.Thread(target=open_browser, args=(port, 1), daemon=True).start()

    print(f"Starting development server on port {port}")
    app.run(debug=debug, port=port, host='127.0.0.1')


if __name__ == "__main__":
    main()