"""
Authentication module for WorldQuant Brain API.
"""
import requests
import logging
import json
import time
from typing import Optional
import sys
import os

logger = logging.getLogger(__name__)

def check_session_valid(session: requests.Session, retry_on_204: bool = False) -> bool:
    """
    Check if the WorldQuant Brain API session is still valid.

    Args:
        session: The authenticated requests.Session object.
        retry_on_204: If True, will retry when receiving 204 status (authentication pending).

    Returns:
        bool: True if the session is valid, False otherwise.
    """
    try:
        auth_url = "https://api.worldquantbrain.com/authentication"
        max_retries = 3 if retry_on_204 else 1
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            result = session.get(auth_url)

            if result.status_code == 200:
                try:
                    expiry = result.json().get("token", {}).get("expiry", 0)
                    logger.info(f"Session valid - token expires in {expiry} seconds.")
                    return expiry > 0
                except json.JSONDecodeError:
                    logger.error("Failed to decode JSON from authentication response.")
                    return False
            elif result.status_code == 204 and retry_on_204 and attempt < max_retries - 1:
                # 204 No Content might mean authentication is still processing
                logger.info(f"Authentication pending (204). Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                continue
            else:
                if result.status_code == 204:
                    logger.warning(f"Session check returned 204 (No Content) - authentication may be pending")
                else:
                    logger.error(f"Session check failed with status code: {result.status_code}. Response: {result.text[:200]}...")
                return False
        
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking session: {e}")
        return False

def get_authenticated_session(session: Optional[requests.Session] = None) -> Optional[requests.Session]:
    """
    Get an authenticated session for the WorldQuant Brain API.
    
    Args:
        session: Optional pre-configured session object. If provided, this session will be authenticated.
                 If None, a new session will be created.
    
    Returns:
        An authenticated requests.Session object or None if authentication fails.
    """
    # Import ace module from project root
    try:
        # First try to import directly (if package is properly installed or in PYTHONPATH)
        import ace
    except ImportError:
        # If direct import fails, try relative import paths
        logger.debug(f"Direct import of ace failed, trying alternative paths")
        current_file_dir = os.path.dirname(os.path.abspath(__file__))  # .../alphaDataBank/api
        project_root = os.path.abspath(os.path.join(current_file_dir, '..'))  # .../alphaDataBank
        
        # Add project root to path only temporarily for this function
        import importlib.util
        try:
            # Try to load the module by file path
            spec = importlib.util.spec_from_file_location("ace", os.path.join(project_root, "ace.py"))
            if spec and spec.loader:
                ace = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ace)
                logger.debug(f"Successfully loaded ace module from {os.path.join(project_root, 'ace.py')}")
            else:
                raise ImportError(f"Could not load ace module from {os.path.join(project_root, 'ace.py')}")
        except Exception as e:
            logger.error(f"Failed to import ace module: {e}")
            raise
    
    # Use the provided session if available, otherwise create a new one
    if session is None:
        session = ace.start_session()
    else:
        # If a session is provided, authenticate it
        session = ace.authenticate_existing_session(session)
    
    # First check without retry to see immediate status
    if check_session_valid(session, retry_on_204=False):
        logger.info("Successfully authenticated with WorldQuant Brain API")
        return session
    
    # If initial check fails, wait a moment and retry with 204 handling
    logger.info("Initial authentication check failed. Waiting for authentication to complete...")
    time.sleep(2)
    
    # Now check with retry logic for 204 responses
    if check_session_valid(session, retry_on_204=True):
        logger.info("Successfully authenticated with WorldQuant Brain API after retry")
        return session
    else:
        logger.error("Failed to authenticate with WorldQuant Brain API after retries")
        return None
