# Alpha DataBank

A PostgreSQL-based database system for tracking and analyzing WorldQuant Brain alpha trading strategies with optimized correlation calculation.

## Overview

The Alpha DataBank system allows you to:

1. **Fetch alpha metadata** from WorldQuant Brain API (submitted & unsubmitted)
2. **Store alpha metrics and PNL data** in a PostgreSQL database
3. **Calculate correlation statistics** between alphas using optimized algorithms
4. **Query and analyze alpha performance** through interactive dashboards
5. **Perform clustering analysis** of alpha strategies and expressions

### ✨ **New Features (2025)**

- **🚀 Automated Unsubmitted Alpha Fetching**: Overcomes 10,000 alpha API limits through intelligent date windowing
- **⚡ Parallel Processing**: 10x faster fetching using ThreadPoolExecutor for both submitted and unsubmitted alphas
- **📊 Interactive Dashboards**: Real-time clustering visualization and analysis interfaces
- **🎯 Smart Authentication**: Automatic retry logic that works on first attempt
- **🔍 Advanced Filtering**: Custom sharpe ratio thresholds and batch size optimization

## Project Structure

```
alphaDataBank/
├── api/                              # API interaction modules
│   ├── auth.py                       # Authentication with WorldQuant Brain API
│   ├── alpha_fetcher.py              # Submitted alphas fetching (parallel)
│   ├── unsubmitted_fetcher.py        # URL-based unsubmitted alphas (parallel)
│   └── unsubmitted_fetcher_auto.py   # Automated unsubmitted alphas (NEW)
├── database/                         # Database modules
│   ├── operations.py                 # Core database operations for alphas and PNL
│   ├── operations_unsubmitted.py     # Unsubmitted alphas database operations
│   └── schema.py                     # Database schema initialization
├── analysis/                         # Analysis and clustering modules
│   ├── clustering/                   # Alpha clustering analysis
│   │   ├── clustering_analysis.py    # Clustering algorithms and analysis
│   │   └── visualization_server.py   # Interactive dashboard (Dash/Plotly)
│   ├── analysis_operations.py        # Core analysis operations
│   └── alpha_expression_parser.py    # Alpha expression parsing and analysis
├── scripts/                          # Executable scripts
│   ├── run_alpha_databank.py         # ⭐ Main orchestration script (USE THIS)
│   ├── init_database.py              # Database initialization
│   ├── calculate_cross_correlation.py # Cross-correlation analysis
│   ├── calculate_unsubmitted_correlations.py # Unsubmitted vs submitted correlations
│   └── update_correlations_optimized.py # Optimized correlation calculation
├── config/                           # Configuration files
│   ├── database_config.py            # Database connection parameters
│   └── api_config.py                 # API configuration and defaults
├── utils/                            # Utility functions
│   ├── helpers.py                    # Common utility functions (logging, reporting)
│   └── correlation_utils.pyx         # Cython-accelerated correlation calculation
├── run_analysis_dashboard.py         # Alternative analysis dashboard
├── AUTOMATED_UNSUBMITTED_FETCHING.md # Documentation for new automated features
└── ace.py                            # WorldQuant Brain authentication module
```

## Setup

1. Make sure PostgreSQL is installed and running on your machine.
2. Update database connection parameters in `config/database_config.py`.
3. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

### 🚀 Quick Start Commands

#### **Main Processing Script**
```bash
# Complete workflow for a specific region (recommended)
./venv/Scripts/python.exe scripts/run_alpha_databank.py --region USA

# Complete workflow for all regions
./venv/Scripts/python.exe scripts/run_alpha_databank.py --all

# Daily updates (skip initialization)
./venv/Scripts/python.exe scripts/run_alpha_databank.py --region USA --skip-init
```

#### **Analysis Dashboard**
```bash
# Launch interactive analysis dashboard
./venv/Scripts/python.exe analysis/clustering/visualization_server.py

# Launch with specific port
./venv/Scripts/python.exe analysis/clustering/visualization_server.py --port 8050
```

#### **Alternative Analysis Dashboard**
```bash
# Launch alternative dashboard
./venv/Scripts/python.exe run_analysis_dashboard.py
```

---

### 📊 **Alpha Fetching Commands**

#### **Submitted Alphas (Regular Processing)**
```bash
# Fetch submitted alphas for specific region
./venv/Scripts/python.exe scripts/run_alpha_databank.py --region USA

# Fetch submitted alphas for all regions
./venv/Scripts/python.exe scripts/run_alpha_databank.py --all

# Skip specific steps
./venv/Scripts/python.exe scripts/run_alpha_databank.py --region USA --skip-pnl-fetch --skip-correlation
```

#### **Unsubmitted Alphas (Manual URL Method)**
```bash
# Fetch unsubmitted alphas using provided URL
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted --url "https://api.worldquantbrain.com/users/self/alphas?limit=50&offset=0&status=UNSUBMITTED%1FIS_FAIL&is.sharpe%3E=%201&dateCreated%3E=2020-08-01T00:00:00-04:00&dateCreated%3C2025-08-25T00:00:00-04:00&order=-dateCreated&hidden=false" --region USA
```

#### **Unsubmitted Alphas (Automated Method) ⭐ NEW**
```bash
# Fetch ALL unsubmitted alphas automatically (recommended)
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --all

# Fetch for specific region with default thresholds (>= 1.0 and <= -1.0)
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --region USA

# Custom sharpe thresholds
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --sharpe-thresholds "2,-2" --region USA

# High-quality alphas only (sharpe >= 2.0)
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --sharpe-thresholds "2" --all

# Multiple thresholds
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --sharpe-thresholds "2,1.5,-1.5,-2" --all

# Larger batch size for faster processing
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --batch-size 100 --region USA
```

---

### 🔧 **Individual Component Commands**

#### **Database Initialization**
```bash
# Initialize main database schema
./venv/Scripts/python.exe scripts/init_database.py

# The main script also handles initialization automatically
./venv/Scripts/python.exe scripts/run_alpha_databank.py --region USA
```

#### **Alpha Metadata Only**
```bash
# Fetch only metadata (skip PNL and correlations)
./venv/Scripts/python.exe scripts/run_alpha_databank.py --region USA --skip-pnl-fetch --skip-correlation

# Legacy individual script (use main script instead)
./venv/Scripts/python.exe scripts/fetch_alphas.py --region USA
```

#### **PNL Data Only**
```bash
# Fetch only PNL data (skip metadata and correlations)
./venv/Scripts/python.exe scripts/run_alpha_databank.py --region USA --skip-alpha-fetch --skip-correlation

# Legacy individual script (use main script instead) 
./venv/Scripts/python.exe scripts/fetch_pnl.py --region USA
```

#### **Correlation Calculations Only**
```bash
# Calculate only correlations (skip fetching)
./venv/Scripts/python.exe scripts/run_alpha_databank.py --region USA --skip-alpha-fetch --skip-pnl-fetch

# Legacy individual script (use main script instead)
./venv/Scripts/python.exe scripts/update_correlations_optimized.py --region USA
```

#### **Generate Reports**
```bash
# Generate correlation report
./venv/Scripts/python.exe scripts/run_alpha_databank.py --region USA --report

# Report for all regions
./venv/Scripts/python.exe scripts/run_alpha_databank.py --all --report
```

---

### 📈 **Analysis & Visualization Commands**

#### **Clustering Analysis**
```bash
# Run clustering analysis for a region
./venv/Scripts/python.exe analysis/clustering/clustering_analysis.py --region USA

# Run for all regions
./venv/Scripts/python.exe analysis/clustering/clustering_analysis.py --all
```

#### **Interactive Dashboards**
```bash
# Main analysis dashboard with clustering visualization
./venv/Scripts/python.exe analysis/clustering/visualization_server.py

# Alternative analysis interface
./venv/Scripts/python.exe run_analysis_dashboard.py

# Specify custom port
./venv/Scripts/python.exe analysis/clustering/visualization_server.py --port 8080
```

#### **Cross-Correlation Analysis**
```bash
# Calculate cross-correlations (advanced)
./venv/Scripts/python.exe scripts/calculate_cross_correlation.py --csv-file EUR_TEMPLATE_SH37.csv --region EUR --limit 5
```

---

### ⚙️ **Command Options Reference**

#### **Main Script Options (`run_alpha_databank.py`)**
| Option | Description | Example |
|--------|-------------|---------|
| `--region REGION` | Process specific region | `--region USA` |
| `--all` | Process all configured regions | `--all` |
| `--skip-init` | Skip database initialization | `--skip-init` |
| `--skip-alpha-fetch` | Skip alpha metadata fetching | `--skip-alpha-fetch` |
| `--skip-pnl-fetch` | Skip PNL data fetching | `--skip-pnl-fetch` |
| `--skip-correlation` | Skip correlation calculation | `--skip-correlation` |
| `--report` | Print correlation report | `--report` |

#### **Unsubmitted Alpha Options**
| Option | Description | Example |
|--------|-------------|---------|
| `--unsubmitted` | Manual URL-based fetching | `--unsubmitted --url "..."` |
| `--unsubmitted-auto` | Automated fetching (recommended) | `--unsubmitted-auto` |
| `--url URL` | URL for manual fetching | `--url "https://..."` |
| `--sharpe-thresholds THRESHOLDS` | Comma-separated sharpe values | `--sharpe-thresholds "1,-1"` |
| `--batch-size SIZE` | Batch size for requests | `--batch-size 100` |

#### **Supported Regions**
- `USA`, `EUR`, `JPN`, `CHN`, `AMR`, `ASI`, `GLB`, `HKG`, `KOR`, `TWN`

---

### 🎯 **Common Workflows**

#### **Daily Production Workflow**
```bash
# 1. Update submitted alphas
./venv/Scripts/python.exe scripts/run_alpha_databank.py --all --skip-init

# 2. Update unsubmitted alphas  
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --all --skip-init

# 3. Launch dashboard for analysis
./venv/Scripts/python.exe analysis/clustering/visualization_server.py
```

#### **Initial Setup Workflow**
```bash
# 1. Complete setup for all regions (first time)
./venv/Scripts/python.exe scripts/run_alpha_databank.py --all

# 2. Fetch all unsubmitted alphas
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --all

# 3. Generate reports
./venv/Scripts/python.exe scripts/run_alpha_databank.py --all --report --skip-alpha-fetch --skip-pnl-fetch --skip-correlation
```

#### **Research & Development Workflow**
```bash
# 1. Fetch high-quality alphas only
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --sharpe-thresholds "2" --region USA

# 2. Run clustering analysis
./venv/Scripts/python.exe analysis/clustering/clustering_analysis.py --region USA  

# 3. Launch visualization dashboard
./venv/Scripts/python.exe analysis/clustering/visualization_server.py
```

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

---

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **Authentication Problems**
```bash
# If authentication fails on first attempt, it should retry automatically
# Check if cookies exist in secrets/ directory
ls secrets/session_cookies.json

# Manual re-authentication (if needed)
rm secrets/session_cookies.json
./venv/Scripts/python.exe scripts/run_alpha_databank.py --region USA
```

#### **Database Connection Issues**
```bash
# Test database connection
./venv/Scripts/python.exe -c "from database.schema import get_connection; print('Connected:', get_connection() is not None)"

# Reinitialize database if needed
./venv/Scripts/python.exe scripts/init_database.py
```

#### **Performance Issues**
```bash
# Use smaller batch sizes if requests timeout
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --batch-size 25 --region USA

# Process single region instead of all
./venv/Scripts/python.exe scripts/run_alpha_databank.py --region USA  # instead of --all
```

#### **Memory Issues with Large Data Sets**
```bash
# Process regions individually
for region in USA EUR JPN CHN; do
  ./venv/Scripts/python.exe scripts/run_alpha_databank.py --region $region --skip-init
done
```

### **Log Files**
- Application logs are displayed in console output
- Check database logs in PostgreSQL logs directory
- Enable debug logging by modifying `utils/helpers.py`

---

## 📋 **Daily Operations**

### **Recommended Daily Workflow**
```bash
# Morning: Update all data
./venv/Scripts/python.exe scripts/run_alpha_databank.py --all --skip-init
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --all --skip-init

# Afternoon: Analysis and review
./venv/Scripts/python.exe analysis/clustering/visualization_server.py
# Open browser to http://localhost:8050
```

### **Weekly Maintenance**
```bash
# Full refresh (includes database cleanup)
./venv/Scripts/python.exe scripts/run_alpha_databank.py --all

# Generate comprehensive reports
./venv/Scripts/python.exe scripts/run_alpha_databank.py --all --report --skip-alpha-fetch --skip-pnl-fetch --skip-correlation
```

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
