"""
Authentication module for WorldQuant Brain API.
"""
import requests
import logging
import json
import time
from typing import Optional, Dict, Any, Tuple, Union
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
            # Use raw session.get here to avoid infinite recursion in authenticated_get
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
    # Import ace module from legacy directory
    try:
        # First try to import directly (if package is properly installed or in PYTHONPATH)
        from legacy import ace
    except ImportError:
        # If direct import fails, try relative import paths
        logger.debug(f"Direct import of ace failed, trying alternative paths")
        current_file_dir = os.path.dirname(os.path.abspath(__file__))  # .../alphaDataBank/api
        project_root = os.path.abspath(os.path.join(current_file_dir, '..'))  # .../alphaDataBank
        
        # Add project root to path only temporarily for this function
        import importlib.util
        try:
            # Try to load the module by file path from legacy directory
            spec = importlib.util.spec_from_file_location("ace", os.path.join(project_root, "legacy", "ace.py"))
            if spec and spec.loader:
                ace = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ace)
                logger.debug(f"Successfully loaded ace module from {os.path.join(project_root, 'legacy', 'ace.py')}")
            else:
                raise ImportError(f"Could not load ace module from {os.path.join(project_root, 'legacy', 'ace.py')}")
        except Exception as e:
            logger.error(f"Failed to import ace module: {e}")
            raise
    
    # Use the provided session if available, otherwise create a new one
    if session is None:
        session = ace.start_session()
    else:
        # If a session is provided, authenticate it
        session = ace.authenticate_existing_session(session)
    
    # Check with retry logic for 204 responses (single phase)
    if check_session_valid(session, retry_on_204=True):
        logger.info("Successfully authenticated with WorldQuant Brain API")
        return session
    else:
        logger.error("Failed to authenticate with WorldQuant Brain API after retries")
        return None


# Global session reference for automatic reauthentication
_current_session: Optional[requests.Session] = None


def _reauthenticate_session_inplace(session: requests.Session) -> bool:
    """
    Reauthenticate a session by updating it in-place (same object reference).
    
    Args:
        session: The session to reauthenticate
        
    Returns:
        True if reauthentication succeeded, False otherwise
    """
    try:
        # Import ace module to use its authentication logic
        from legacy import ace
        
        # Clear the existing authentication and cookies
        session.auth = None
        session.cookies.clear()
        
        # Use ace's authenticate_existing_session to update the session in-place
        authenticated_session = ace.authenticate_existing_session(session)
        
        # The session object is the same, but now it should have fresh cookies/auth
        # Verify it worked
        if authenticated_session is session and check_session_valid(session, retry_on_204=False):
            return True
        else:
            logger.error("In-place reauthentication failed - session validation failed")
            return False
            
    except Exception as e:
        logger.error(f"In-place reauthentication failed: {e}")
        return False


def authenticated_request(
    method: str, 
    url: str, 
    session: Optional[requests.Session] = None,
    max_auth_retries: int = 2,
    **kwargs
) -> requests.Response:
    """
    Make an authenticated HTTP request with automatic reauthentication on 401 errors.
    
    This function wraps all HTTP requests and automatically handles authentication
    failures by reauthenticating the SAME session object in-place.
    
    Args:
        method: HTTP method ('GET', 'POST', etc.)
        url: Target URL
        session: Optional session to use. If None, uses global session.
        max_auth_retries: Maximum number of reauthentication attempts
        **kwargs: Additional arguments passed to requests (params, headers, timeout, etc.)
    
    Returns:
        requests.Response: The HTTP response
        
    Raises:
        requests.exceptions.RequestException: If request fails after all retries
        Exception: If reauthentication fails completely
    """
    global _current_session
    
    # Determine which session to use
    target_session = session if session is not None else _current_session
    
    # If no session available, create one
    if target_session is None:
        logger.info("No session available, creating new authenticated session...")
        target_session = get_authenticated_session()
        if target_session is None:
            raise Exception("Failed to create authenticated session")
        if session is None:  # Only update global if we're using global
            _current_session = target_session
    
    auth_attempt = 0
    while auth_attempt <= max_auth_retries:
        try:
            # Make the request
            response = target_session.request(method, url, **kwargs)
            
            # Check for authentication errors
            if response.status_code == 401:
                auth_attempt += 1
                logger.warning(f"Authentication error (401) on attempt {auth_attempt}. Response: {response.text[:200]}")
                
                if auth_attempt <= max_auth_retries:
                    logger.info(f"Attempting in-place reauthentication (attempt {auth_attempt}/{max_auth_retries})...")
                    
                    # Reauthenticate the SAME session object in-place
                    if _reauthenticate_session_inplace(target_session):
                        logger.info(f"In-place reauthentication successful, retrying request...")
                        continue  # Retry the request with the updated session
                    else:
                        logger.error(f"In-place reauthentication failed on attempt {auth_attempt}")
                        continue
                else:
                    logger.error(f"Exceeded maximum authentication retries ({max_auth_retries})")
                    raise requests.exceptions.HTTPError(f"Authentication failed after {max_auth_retries} retries", response=response)
            
            # Request successful (not a 401), return response
            return response
            
        except requests.exceptions.RequestException as e:
            # Non-auth related request errors - let caller handle them
            if auth_attempt == 0:  # Only log on first attempt to avoid spam
                logger.debug(f"Request error: {e}")
            raise e
    
    # This shouldn't be reached, but just in case
    raise Exception(f"Unexpected end of authenticated_request after {auth_attempt} attempts")


def authenticated_get(url: str, session: Optional[requests.Session] = None, **kwargs) -> requests.Response:
    """Make an authenticated GET request with automatic reauthentication."""
    return authenticated_request('GET', url, session=session, **kwargs)


def authenticated_post(url: str, session: Optional[requests.Session] = None, **kwargs) -> requests.Response:
    """Make an authenticated POST request with automatic reauthentication."""
    return authenticated_request('POST', url, session=session, **kwargs)


def set_global_session(session: requests.Session) -> None:
    """Set the global session for authenticated requests."""
    global _current_session
    _current_session = session
    logger.debug("Global session updated")


def get_global_session() -> Optional[requests.Session]:
    """Get the current global session."""
    return _current_session
