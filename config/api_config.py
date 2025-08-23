# WorldQuant Brain API Configuration
BASE_API_URL = "https://api.worldquantbrain.com"
DEFAULT_REQUEST_LIMIT = 50
AUTHENTICATION_URL = f"{BASE_API_URL}/authentication"
ALPHAS_ENDPOINT = f"{BASE_API_URL}/users/self/alphas" # Base for fetching alphas
ALPHA_PNL_ENDPOINT_TEMPLATE = f"{BASE_API_URL}/alphas/{{alpha_id}}/recordsets/pnl" # Template for PNL

# Default parameters for fetching alphas
# The status parameter might need adjustment based on API specifics.
# The original URL used 'status!=UNSUBMITTED%1FIS-FAIL'.
# '%1F' is Unit Separator. If the API expects this literally, it's unusual.
# A common way is multiple 'status!=' params or a specific format.
# For now, keeping it as a string that might need to be URL-encoded or split.
DEFAULT_ALPHA_FETCH_PARAMS = {
    "limit": "50", # API expects string for limit and offset
    "status!": "UNSUBMITTED,IS-FAIL", # Placeholder, might need to be "UNSUBMITTED%1FIS-FAIL" or handled specially
    "order": "-dateCreated",
    "hidden": "false"
}

# Rate limiting (in seconds)
DEFAULT_RETRY_AFTER_SECONDS = 1.0 # Default sleep time if 'Retry-After' header is present
API_REQUEST_DELAY_SECONDS = 1.0 # General delay between PNL API calls to be polite
