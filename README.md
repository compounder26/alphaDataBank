# Alpha DataBank

A PostgreSQL-based database system for tracking and analyzing WorldQuant Brain alpha trading strategies with optimized correlation calculation.

## Overview

The Alpha DataBank system allows you to:

1. Fetch alpha metadata from WorldQuant Brain API
2. Store alpha metrics and PNL data in a PostgreSQL database
3. Calculate correlation statistics between alphas
4. Query and analyze alpha performance

## Project Structure

```
alphaDataBank/
├── config/                           # Configuration files
│   └── database_config.py            # Database connection parameters
├── database/                         # Database modules
│   └── operations.py                 # Core database operations for alphas and PNL
├── api/                              # API interaction modules
│   └── auth.py                       # Authentication with WorldQuant Brain API
├── utils/                            # Utility functions
│   ├── helpers.py                    # Common utility functions (logging, reporting)
│   └── correlation_utils.pyx         # Cython-accelerated correlation calculation
└── scripts/                          # Executable scripts
    ├── init_database.py              # Database initialization
    ├── fetch_alphas.py               # Fetch alpha metadata from API
    ├── fetch_pnl.py                  # Fetch PNL data for alphas
    ├── update_correlations.py        # Original correlation calculation
    ├── update_correlations_optimized.py  # Optimized correlation with Cython
    └── run_alpha_databank.py         # Main orchestration script
```

## Setup

1. Make sure PostgreSQL is installed and running on your machine.
2. Update database connection parameters in `config/database_config.py`.
3. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

### Initialize the Database

Create the necessary database and tables:

```
python scripts/init_database.py
```

### Fetch Alpha Data

Fetch alpha metadata for a specific region:

```
python scripts/fetch_alphas.py --region USA
```

Or fetch for all regions:

```
python scripts/fetch_alphas.py --all
```

### Fetch PNL Data

Fetch PNL data for alphas in a specific region:

```
python scripts/fetch_pnl.py --region USA
```

Or for a specific alpha:

```
python scripts/fetch_pnl.py --region USA --alpha-id YOUR_ALPHA_ID
```

### Calculate Correlations

Calculate and update correlation statistics for alphas in a region:

```
python scripts/update_correlations.py --region USA
```

### Run Complete Process

To run the entire process (initialize, fetch alphas, fetch PNL, calculate correlations):

```
python scripts/run_alpha_databank.py --region USA
```

Optional flags:
- `--skip-init`: Skip database initialization
- `--skip-alpha-fetch`: Skip alpha fetching
- `--skip-pnl-fetch`: Skip PNL fetching
- `--skip-correlation`: Skip correlation calculation
- `--report`: Print correlation report

## Database Schema

1. **regions**: Stores information about different regions
   - `region_id`: Unique identifier for each region
   - `region_name`: Name of the region (e.g., 'USA', 'ASIA', etc.)

2. **alphas**: Stores metadata about each alpha
   - `alpha_id`: Unique identifier for each alpha strategy
   - `region_id`: Foreign key referencing the regions table
   - `title`: Alpha title/name
   - `creator`: Alpha creator's username
   - `creation_date`: When the alpha was created
   - `last_update_date`: When the alpha was last updated
   - `return`: Average daily return
   - `sharpe`: Sharpe ratio
   - `fitness`: Overall fitness score
   - `mdd`: Maximum drawdown
   - `correlation`: Correlation (self-correlation or custom benchmark)
   - Various other metrics and properties

3. **pnl_[region]**: Region-specific tables for PNL data
   - `alpha_id`: Alpha identifier
   - `date`: Trading date
   - `pnl`: Profit and Loss value
   - Composite primary key (alpha_id, date)

4. **correlation_[region]**: Region-specific tables for correlation statistics
   - `alpha_id`: Alpha identifier
   - `min_corr`: Minimum correlation with other alphas
   - `max_corr`: Maximum correlation with other alphas
   - `avg_corr`: Average correlation with other alphas
   - `median_corr`: Median correlation with other alphas

## Authentication

The system uses `ace.start_session()` for authentication with the WorldQuant Brain API. The ace module should be located at the project root as 'ace.py'. The authentication module uses importlib to dynamically load this module from the project root.

## Daily Updates

For daily updates, you can run:

```
python scripts/run_alpha_databank.py --region USA --skip-init
```

This will fetch any new alphas, update PNL data, and recalculate correlations.

## Performance Optimizations

### Smart Correlation Skipping

The system tracks whether new alpha metadata or PNL data has been inserted for each region during the current run. If no new data has been added for a region, correlation calculations for that region will be skipped, improving performance significantly for daily updates.

### Cython-Accelerated Correlations

The project includes a Cython-optimized implementation of the correlation calculation:

1. Utilizes compiled C code for faster numerical operations
2. Includes a pure Python fallback if Cython compilation fails
3. Processes correlations in parallel using Python's multiprocessing

To build the Cython extension, run:

```
python scripts/build_cython_modules.py build_ext --inplace
```

### Optimized Database Operations

The system uses PostgreSQL's COPY command for efficient bulk inserts of PNL data, significantly improving data loading performance compared to individual INSERT statements.
