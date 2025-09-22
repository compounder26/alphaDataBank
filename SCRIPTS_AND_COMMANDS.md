# Alpha DataBank - Scripts and Commands Reference

## Table of Contents

1. [Data Management Scripts](#data-management-scripts)
2. [Dashboard and Analysis Commands](#dashboard-and-analysis-commands)
3. [Cache and Data Management](#cache-and-data-management)
4. [Database Utility Scripts](#database-utility-scripts)
5. [Production Deployment Commands](#production-deployment-commands)
6. [Configuration Examples](#configuration-examples)
7. [Performance Optimization Code](#performance-optimization-code)
8. [Common Workflows](#common-workflows)

---

## Data Management Scripts

### Primary Data Fetching

**`run_alpha_databank.py`** - Main data pipeline script

```bash
# Fetch all submitted alphas for all regions
python run_alpha_databank.py --all

# Fetch data for specific region
python run_alpha_databank.py --region USA

# Fetch unsubmitted alphas from URL
python run_alpha_databank.py --unsubmitted --url "https://api.worldquantbrain.com/users/self/alphas?..." --all

```

**Key parameters**:
- `--all`: Fetch data for all configured regions
- `--region`: Specify single region (USA, EUR, CHN, ASI, GLB, etc.)
- `--unsubmitted`: Fetch unsubmitted alphas

---

## Dashboard and Analysis Commands

**`run_analysis_dashboard.py`** - Interactive dashboard

```bash
# Run dashboard in development mode
python run_analysis_dashboard.py

# Run on different port
python run_analysis_dashboard.py --port 8051

# Refresh operators/datafields and clear cache
python run_analysis_dashboard.py --renew

# Clear analysis cache only
python run_analysis_dashboard.py --clear-cache
```

---

## Cache and Data Management

**`clear_cache.py`** - Clear analysis caches

```bash
# Clear all analysis caches
python clear_cache.py

```

**`renew_genius.py`** - Refresh operator/datafield data

```bash
# Refresh operators and datafields from API
python renew_genius.py

```

**`refresh_clustering.py`** - Regenerate clustering analysis

```bash
# Refresh clustering for all regions
python refresh_clustering.py

# Refresh specific regions only
python refresh_clustering.py --regions USA EUR CHN

```

---

## Database Utility Scripts

**`scripts/init_database.py`** - Initialize database schema

```bash
# Initialize main database
python scripts/init_database.py

```

**`scripts/calculate_correlations.py`** - Manual correlation calculation

```bash
# Calculate submitted alpha correlations for all regions
python scripts/calculate_correlations.py --mode submitted --all-regions

# Calculate submitted correlations for specific region
python scripts/calculate_correlations.py --mode submitted --region USA

# Calculate unsubmitted vs submitted correlations
python scripts/calculate_correlations.py --mode unsubmitted --region USA

# Cross-correlation analysis with CSV export
python scripts/calculate_correlations.py --mode cross --alpha-ids alpha1,alpha2,alpha3 --csv-export results.csv
```

---

## Production Deployment Commands

**Development Mode**:
```bash
python run_analysis_dashboard.py
```

**Production Mode - Windows**:
```bash
waitress-serve --host=127.0.0.1 --port=8050 wsgi:server
```

**Production Mode - Unix/Linux/Mac**:
```bash
gunicorn -w 4 -b 127.0.0.1:8050 wsgi:server
```


---

## Configuration Examples

### Environment Variables (`.env`)

```env
# Database Configuration
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=alpha_database

# API Configuration
BRAIN_API_URL=https://api.worldquantbrain.com
API_RATE_LIMIT=10
API_TIMEOUT=30

# Cache Configuration
CACHE_EXPIRY_HOURS=24
CACHE_MAX_SIZE_MB=1000

# Performance Settings
MAX_WORKERS=8
CORRELATION_BATCH_SIZE=1000
```


---

## Performance Optimization Code

### Cython Optimization

**Compilation Setup** (`setup.py`):
```python
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "correlation_utils",
        ["correlation_utils.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

setup(
    name="correlation_utils",
    ext_modules=cythonize(
        extensions,
        language_level=3,
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'profile': False
        }
    )
)
```

**Check Cython Status**:
```bash
python utils/cython_helper.py
```

### Database Optimization

**Batch PNL Data Insertion**:
```python
def insert_multiple_pnl_data_optimized(pnl_data: List[Tuple], region: str) -> None:
    """
    Optimized batch PNL insertion using PostgreSQL COPY.
    Performance: ~100x faster than individual inserts
    """
    table_name = f"pnl_{region.lower()}"

    buffer = io.StringIO()
    for alpha_id, date, pnl in pnl_data:
        buffer.write(f"{alpha_id}\t{date}\t{pnl}\n")
    buffer.seek(0)

    conn = psycopg2.connect(**connection_params)
    cursor = conn.cursor()
    cursor.copy_from(
        buffer,
        table_name,
        columns=['alpha_id', 'date', 'pnl'],
        sep='\t'
    )
    conn.commit()
```

### Caching Implementation

The project uses a simple but effective two-tier caching system:

**1. File System Cache (API Data)**
```bash
# Cache files are automatically created in the data/ directory:
data/operators_dynamic.json     # Cached operators from API
data/datafields_dynamic.csv     # Cached datafields from API
data/.cache_metadata.json       # Cache timestamps and metadata
```

**2. Database Cache (Analysis Results)**
```sql
-- Analysis results are cached in PostgreSQL table:
SELECT * FROM alpha_analysis_cache;  -- Parsed alpha analysis results
```

**How it works:**
- `python renew_genius.py` → Fetches fresh data from API → Saves to JSON/CSV files → Stores datafields in database
- `python clear_cache.py` → Clears analysis cache table → Forces dashboard to re-analyze all alphas
- Dashboard loads operators from JSON file and datafields from database
- Analysis results are cached in database to avoid re-parsing alpha expressions

**Cache locations:**
- **API cache**: `data/operators_dynamic.json`, `data/datafields_dynamic.csv`
- **Analysis cache**: `alpha_analysis_cache` database table
- **Clustering cache**: `data/clustering_*.json` files (auto-generated)

---

## Common Workflows

### Initial Setup Workflow

```bash
# 1. Initialize database
python scripts/init_database.py

# 2. Fetch alpha data
python run_alpha_databank.py --all

# 3. Generate clustering analysis
python refresh_clustering.py

# 4. Run dashboard
python run_analysis_dashboard.py
```

### Regular Maintenance Workflow

```bash
# 1. Refresh operators/datafields (every genius tier update)
python renew_genius.py

# 2. Fetch new alpha data (every new alpha submitted)
python run_alpha_databank.py --all

# 3. Refresh clustering (every new submitted alpha in the database)
python refresh_clustering.py

# 4. Clear old caches (every new submitted alpha in the database)
python clear_cache.py
```