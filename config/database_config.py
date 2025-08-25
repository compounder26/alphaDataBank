import os
from dotenv import load_dotenv

# Load environment variables from .env file
# Ensure you have a .env file in the project root (alphaDataBank/.env)
# Example .env content:
# DB_USER=your_user
# DB_PASSWORD=your_password
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=alpha_database # Your current database name
load_dotenv()

# PostgreSQL connection details
DB_USER = os.getenv("DB_USER", "postgres") # Default to 'postgres' if not set
DB_PASSWORD = os.getenv("DB_PASSWORD")  # REQUIRED: Set in .env file for security
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "alpha_database") # Default to your current DB name

# Validate required environment variables
if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD environment variable must be set in .env file")

# Construct DATABASE_URL for SQLAlchemy
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Available regions for alphas (preserved from your existing config)
REGIONS = ['USA', 'EUR', 'JPN', 'CHN', 'AMR', 'ASI', 'GLB', 'HKG', 'KOR', 'TWN']
