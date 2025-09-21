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
from .services import load_clustering_data, load_operators_data
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
    parser.add_argument("--data-file", type=str, help="Path to the clustering data JSON file (optional)")
    parser.add_argument("--port", type=int, default=DASHBOARD_SETTINGS['default_port'], help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--init-db", action="store_true", help="Initialize analysis database schema")
    parser.add_argument("--operators-file", type=str, help="Path to custom operators file (JSON or TXT)")

    args = parser.parse_args()

    # Set global file paths based on arguments (preserve original logic)
    if args.operators_file:
        set_operators_file(args.operators_file)
        print(f"Using dynamic operators file: {args.operators_file}")
    else:
        set_operators_file(DEFAULT_OPERATORS_FILE)
        print(f"Using default operators file: {DEFAULT_OPERATORS_FILE}")

    # Load dynamic operators list if using JSON file (preserve original logic)
    operators_file = args.operators_file or DEFAULT_OPERATORS_FILE
    if operators_file.endswith('.json'):
        try:
            operators_list = load_operators_data(operators_file)
            set_dynamic_operators_list(operators_list)
            print(f"✅ Loaded {len(operators_list)} dynamic operators from JSON file")
        except Exception as e:
            print(f"⚠️ Error loading operators from JSON: {e}, falling back to file parsing")

    # Initialize database if requested (preserve original logic)
    if args.init_db:
        print("Initializing analysis database schema...")
        try:
            initialize_analysis_database()
            print("Analysis database schema initialized successfully.")
        except Exception as e:
            print(f"Error initializing database: {e}")
            return

    # Load clustering data if provided (preserve original logic)
    data = None
    if args.data_file:
        if not os.path.isfile(args.data_file):
            print(f"Warning: Clustering data file not found: {args.data_file}")
            print("Starting in analysis-only mode...")
        else:
            try:
                data = load_clustering_data(args.data_file)
                print(f"Loaded clustering data for region: {data.get('region', 'Unknown')}")
            except Exception as e:
                print(f"Warning: Could not load clustering data: {e}")
                print("Starting in analysis-only mode...")

    # Create app using refactored factory
    print("Initializing dashboard...")
    app = create_visualization_app(data, getattr(sys.modules[__name__], 'DYNAMIC_OPERATORS_LIST', None))

    # Start browser opener thread (preserve original logic)
    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port, DASHBOARD_SETTINGS['browser_delay_seconds']), daemon=True).start()

    # Run server (preserve original logic)
    mode = "Analysis & Clustering" if data else "Analysis-only"
    print(f"Starting {mode} dashboard on port {args.port}...")
    print(f"Dashboard URL: http://localhost:{args.port}")

    if not args.no_browser:
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