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
import getpass
from pathlib import Path
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter

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
    auth_url = "https://api.worldquantbrain.com/authentication"
    max_retries = 5 if retry_on_204 else 1
    # Progressive delays for better handling of token propagation
    retry_delays = [3, 5, 7, 10, 15] if retry_on_204 else [2]

    server_error_retries = 3
    server_error_base_delay = 2

    for attempt in range(max_retries):
        # Inner loop for server error retries
        for server_attempt in range(server_error_retries):
            try:
                # Use raw session.get here to avoid infinite recursion in authenticated_get
                result = session.get(auth_url, timeout=30)

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
                    delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                    logger.info(f"Authentication pending (204). Token propagation in progress. Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    break  # Break from server error loop, continue with main loop
                elif result.status_code >= 500:
                    # Server error - retry with exponential backoff
                    if server_attempt < server_error_retries - 1:
                        delay = server_error_base_delay * (2 ** server_attempt)
                        logger.warning(f"Server error {result.status_code}, retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue  # Continue server error retry loop
                    else:
                        logger.error(f"Server error {result.status_code} after {server_error_retries} attempts")
                        return False
                else:
                    if result.status_code == 204:
                        logger.warning(f"Session check returned 204 (No Content) - authentication token not yet available after {max_retries} attempts")
                    else:
                        logger.error(f"Session check failed with status code: {result.status_code}. Response: {result.text[:200]}...")
                    return False
            except requests.exceptions.RequestException as e:
                if "too many 500 error responses" in str(e) or "too many 504 error responses" in str(e):
                    # Special handling for too many server errors
                    if server_attempt < server_error_retries - 1:
                        delay = server_error_base_delay * (2 ** server_attempt)
                        logger.warning(f"Too many server errors: {e}, retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Too many server errors after {server_error_retries} attempts: {e}")
                        return False
                elif server_attempt < server_error_retries - 1:
                    delay = server_error_base_delay * (2 ** server_attempt)
                    logger.warning(f"Request error: {e}, retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Error checking session after {server_error_retries} attempts: {e}")
                    return False

            # If we get here without breaking, break from server error loop
            break

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
    # Use local authentication functions (moved from legacy)
    
    # Use the provided session if available, otherwise create a new one
    if session is None:
        session = start_session()
    else:
        # If a session is provided, authenticate it
        session = authenticate_existing_session(session)
    
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
        # Clear the existing authentication and cookies
        session.auth = None
        session.cookies.clear()

        # Use local authenticate_existing_session to update the session in-place
        authenticated_session = authenticate_existing_session(session)
        
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


# =============================================================================
# Legacy Authentication Functions (moved from legacy/ace.py)
# =============================================================================

# Constants for legacy functions
brain_api_url = os.environ.get("BRAIN_API_URL", "https://api.worldquantbrain.com")

# Define the path to the 'secrets' folder in the project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is api/
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # Go up one level to project root
SECRETS_DIR = os.path.join(PROJECT_ROOT, "secrets")
COOKIE_FILE_PATH = os.path.join(SECRETS_DIR, "session_cookies.json")
CREDENTIALS_FILE_PATH = os.path.join(SECRETS_DIR, "platform-brain.json")


def save_cookies(session):
    """Save session cookies to file."""
    os.makedirs(SECRETS_DIR, exist_ok=True)  # Ensure 'secrets' directory exists
    with open(COOKIE_FILE_PATH, 'w') as f:
        json.dump(session.cookies.get_dict(), f)
    print(f"Cookies saved to {COOKIE_FILE_PATH}")


def load_cookies(session):
    """Load session cookies from file."""
    if Path(COOKIE_FILE_PATH).exists():
        with open(COOKIE_FILE_PATH, 'r') as f:
            cookies = json.load(f)
            session.cookies.update(cookies)
        print(f"Cookies loaded from {COOKIE_FILE_PATH}")


def get_credentials():
    """
    Function to get JSON with platform credentials if they exist, or prompts for new ones.
    """
    if Path(CREDENTIALS_FILE_PATH).exists() and os.path.getsize(CREDENTIALS_FILE_PATH) > 2:
        with open(CREDENTIALS_FILE_PATH) as file:
            data = json.load(file)
    else:
        os.makedirs(SECRETS_DIR, exist_ok=True)  # Ensure 'secrets' directory exists
        email = input("Email:\n")
        password = getpass.getpass(prompt="Password:")
        data = {"email": email, "password": password}
        with open(CREDENTIALS_FILE_PATH, "w") as file:
            json.dump(data, file)
    return data["email"], data["password"]


def check_session_timeout(session):
    """
    Function checks if the session is still valid.
    Returns the remaining time before expiration.
    """
    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = session.get(brain_api_url + "/authentication", timeout=30)

            # Handle different response codes
            if response.status_code == 200:
                result = response.json()
                expiry = result.get("token", {}).get("expiry", 0)
                print(f"Session timeout check: {expiry} seconds remaining")
                return expiry
            elif response.status_code == 204:
                # 204 No Content - authentication might be pending
                print("Session check returned 204 (authentication pending)")
                return 0
            elif response.status_code == 401:
                print("Session expired (401 Unauthorized)")
                return 0
            elif response.status_code >= 500:
                # Server error - retry with exponential backoff
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Server error {response.status_code}, retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"Server error {response.status_code} after {max_retries} attempts")
                    return 0
            else:
                print(f"Unexpected status code: {response.status_code}")
                return 0
        except json.JSONDecodeError:
            print("Failed to decode JSON response from authentication check")
            return 0
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Request error: {e}, retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                print(f"Error checking session timeout after {max_retries} attempts: {e}")
                return 0
        except Exception as e:
            print(f"Unexpected error checking session timeout: {e}")
            return 0

    return 0


def authenticate_existing_session(s):
    """
    Function to authenticate an existing session object.
    This is useful when you want to use a pre-configured session with custom settings.

    Args:
        s: An existing requests.Session object with custom configuration

    Returns:
        The authenticated session object
    """
    load_cookies(s)  # Load cookies if available

    # Check if the session cookies are still valid
    timeout_remaining = check_session_timeout(s)
    if timeout_remaining > 0:
        print("Reusing existing session...")
        return s

    # If the session is not valid, proceed with new authentication
    s.auth = get_credentials()
    r = s.post(brain_api_url + "/authentication")

    if r.status_code == requests.status_codes.codes.unauthorized:
        if r.headers.get("WWW-Authenticate") == "persona":
            print(
                "Complete biometrics authentication and press any key to continue: \n"
                + urljoin(r.url, r.headers["Location"]) + "\n"
            )
            input()
            s.post(urljoin(r.url, r.headers["Location"]))

            while True:
                if s.post(urljoin(r.url, r.headers["Location"])).status_code != 201:
                    input("Biometrics authentication is not complete. Please try again and press any key when completed \n")
                else:
                    break

            # After biometric auth completes, wait for token to be ready
            print("\nBiometric authentication completed. Waiting for token...")
            max_wait = 60  # Maximum 60 seconds to wait
            wait_interval = 2  # Check every 2 seconds
            elapsed = 0

            while elapsed < max_wait:
                timeout_remaining = check_session_timeout(s)
                if timeout_remaining > 0:
                    print(f"Token ready! Session active with {timeout_remaining} seconds remaining.")
                    save_cookies(s)
                    return s

                time.sleep(wait_interval)
                elapsed += wait_interval
                if elapsed % 10 == 0:  # Print progress every 10 seconds
                    print(f"Still waiting for token... ({elapsed}s elapsed)")

            # If we get here, token didn't become available in time
            print(f"Warning: Token not available after {max_wait} seconds.")
            print("Saving cookies anyway - try running the command again.")
            save_cookies(s)
            return s

        else:
            print("\nIncorrect email or password\n")
            os.remove(CREDENTIALS_FILE_PATH)  # Clear saved credentials
            # Use recursion to retry authentication with new credentials
            s.auth = get_credentials()
            return authenticate_existing_session(s)

    # For non-biometric auth (or if no auth needed), just save cookies and return
    save_cookies(s)
    return s


def start_session():
    """
    Function to sign in to the platform and return a session object.
    Reuses saved cookies if they are valid.
    """
    s = requests.Session()
    # Configure HTTPAdapter for increased connection pool size
    adapter = HTTPAdapter(pool_connections=250, pool_maxsize=250)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    return authenticate_existing_session(s)  # Use the common authentication function
