"""
WSGI entry point for production deployment.

This module provides a production-ready entry point for serving the Alpha DataBank
dashboard using WSGI servers like Gunicorn or Waitress.

NOW USES THE REFACTORED MODULAR DASHBOARD ARCHITECTURE.

Usage:
    For Unix/Linux/Mac with Gunicorn:
        gunicorn -w 4 -b 127.0.0.1:8050 wsgi:server

    For Windows with Waitress:
        waitress-serve --port=8050 wsgi:server
"""

import sys
import os

# Setup project path
from utils.bootstrap import setup_project_path
setup_project_path()

# Import the Dash app creation function from refactored dashboard
from analysis.dashboard.app import create_app

# Create the Dash app
app = create_app()

# Expose the Flask server for WSGI
server = app.server

if __name__ == "__main__":
    # This allows testing the WSGI entry point directly
    print("Starting production server on http://127.0.0.1:8050")
    print("Note: This is still using the development server.")
    print("For production, use: gunicorn -w 4 -b 127.0.0.1:8050 wsgi:server")
    app.run_server(debug=False, port=8050, host='127.0.0.1')