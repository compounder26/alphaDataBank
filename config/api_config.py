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

# Datafields fetching configuration
DATAFIELDS_ENDPOINT = f"{BASE_API_URL}/data-fields"
DATAFIELDS_BATCH_SIZE = 50  # Number of datafields to fetch per API request (API max limit)
DATAFIELDS_MAX_WORKERS = 200  # Maximum threads for parallel fetching in comprehensive mode
DATAFIELDS_RETRY_WAIT = 15  # Base retry wait time in seconds for failed requests
DATAFIELDS_REQUEST_TIMEOUT = 20  # HTTP request timeout in seconds
DATAFIELDS_MAX_BACKOFF_SECONDS = 300  # Maximum backoff time (5 minutes)
DATAFIELDS_MAX_RETRIES = None  # None = indefinite retries for transient errors
DATAFIELDS_RATE_LIMIT_WAIT = 60  # Wait time for 429 rate limit responses

# Datafields fetch parameter combinations
DATAFIELDS_DATA_TYPES = ['MATRIX', 'VECTOR', 'GROUP']  # Available data types
DATAFIELDS_REGIONS = ['AMR', 'ASI', 'CHN', 'EUR', 'GLB', 'HKG', 'JPN', 'KOR', 'TWN', 'USA']  # Available regions
DATAFIELDS_UNIVERSES = [
    'TOP5', 'TOP10', 'TOP20', 'TOP50', 'TOP100', 'TOP200', 'TOP400', 'TOP500',
    'TOP600', 'TOP800', 'TOP1000', 'TOP1200', 'TOP1600', 'TOP2000U', 'TOP2500',
    'TOP3000', 'TOPDIV3000', 'TOPSP500', 'MINVOL1M', 'ILLIQUID_MINVOL1M'
]  # Available universes
DATAFIELDS_DELAYS = [0, 1]  # Available delay values

# Offset limit handling configuration
DATAFIELDS_MAX_OFFSET = 9900  # Stay below 10,000 API limit (safety buffer)
DATAFIELDS_ENABLE_FALLBACK = True  # Enable dataset-based fallback when offset limit is reached
DATAFIELDS_FALLBACK_STRATEGIES = ['datasets', 'search']  # Available fallback methods
