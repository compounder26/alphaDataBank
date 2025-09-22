# Alpha DataBank - Technical Documentation

## Table of Contents

1. [Project Overview & Purpose](#project-overview--purpose)
2. [System Architecture](#system-architecture)
3. [Technical Stack](#technical-stack)
4. [Repository Structure](#repository-structure)
5. [Core Components](#core-components)
6. [Data Flow & Processing](#data-flow--processing)
7. [Database Design](#database-design)
8. [Performance Architecture](#performance-architecture)
9. [Security & Authentication](#security--authentication)
10. [Deployment Architecture](#deployment-architecture)

---

## Project Overview & Purpose

Alpha DataBank is a PostgreSQL-based system designed for tracking, analyzing, and visualizing WorldQuant Brain alpha trading strategies. This system serves as a set of tools to help consultants analyse their alphas better through analytics and interactive visualizations.

### What This Project Solves

**Primary Problem**: Consultants need a way to:
- Track and analyze a lot of alphas across multiple regions - to understand how their current diversity stands and suggest new alpha ideas (from operators/templates or datafields recommendations)
- Analyze correlations between different alphas
- Understand operator and datafield usage patterns
- Identify clustering patterns and similarities between alphas
- Filter through unsubmitted alphas by calculating self-correlation locally because self-correlation requests from the API are limited

### Use Cases

1. **Operator and Datafield usage**: Understand which operators and datafields they had used and the concentration or diversity of it
2. **Correlation & Clustering Analysis**: Identify similiar alphas suggested by the correlation calcualtion or the clustering based on the alpha's features
3. **Datafield Recommendations**:   Discover datafields used in your submitted alphas and identify regions where they could be expanded.
4. **Filter through unsubmitted alphas**: Filtering by Sharpe or fitness alone is usually not enough to find the needle in the haystack that is the unsubmitted alpha pool. Filtering using self-correlation might help.

---

## System Architecture

### High-Level Architecture

The system follows a modular, layered architecture:

```
┌─────────────────────────────────────────────────────┐
│                 Dashboard Layer                      │
│         (Dash + Plotly Visualizations)             │
├─────────────────────────────────────────────────────┤
│                Analysis Layer                        │
│     (Clustering, Correlation, Expression Parsing)   │
├─────────────────────────────────────────────────────┤
│                  API Layer                          │
│          (WorldQuant Brain Integration)             │
├─────────────────────────────────────────────────────┤
│                Database Layer                        │
│              (PostgreSQL + SQLAlchemy)             │
└─────────────────────────────────────────────────────┘
```

### Component Interaction

The system operates through several key interactions:

1. **Data Ingestion**: API layer fetches alpha data from WorldQuant Brain
2. **Data Processing**: Analysis layer processes expressions and calculates correlations
3. **Data Storage**: Database layer manages persistent storage and retrieval
4. **Data Visualization**: Dashboard layer provides interactive analytics interface

---

## Technical Stack

### Core Technologies

**Backend Framework & Database**:
- **Python 3.8+**: Primary programming language with modern async/await support
- **PostgreSQL**: Primary database with advanced indexing and JSONB support
- **SQLAlchemy**: Database ORM with connection pooling and query optimization
- **psycopg2**: High-performance PostgreSQL adapter

**Data Processing & Analytics**:
- **Pandas**: Data manipulation and analysis with optimized memory usage
- **NumPy**: Numerical computing foundation for all calculations
- **Cython**: For improvement on performance correlation calculations
- **scikit-learn**: For clustering and dimensionality reduction
- **UMAP**: For dimensionality reduction for clustering visualization
- **HDBSCAN**: Density-based clustering algorithm for alpha similarity

**Web Interface & Visualization**:
- **Dash**: Interactive web application framework with reactive components
- **Plotly**: Interactive plotting library with WebGL acceleration
- **Dash Bootstrap Components**: Professional UI components
- **NetworkX**: Network analysis for alpha relationships

**Production & Deployment**:
- **Gunicorn**: WSGI HTTP server for Unix/Linux/Mac deployment
- **Waitress**: WSGI server for Windows-compatible deployment
- **WSGI**: Standard interface for production deployment

### Version Requirements

The system requires modern versions of dependencies to ensure compatibility:
- Python 3.8+ (3.10+ recommended for performance)
- PostgreSQL 12+ (14+ recommended for JSONB improvements)

---

## Repository Structure

### Directory Organization

```
alphaDataBank/                      # Project root
├── analysis/                      # Analysis and visualization layer
│   ├── clustering/                # Clustering algorithms and utilities
│   ├── correlation/               # Correlation analysis engine
│   ├── dashboard/                 # Modular dashboard architecture
│   └── alpha_expression_parser.py # Alpha code parsing
├── api/                           # External API integration
│   ├── auth.py                   # WorldQuant Brain authentication
│   ├── alpha_fetcher.py          # Alpha data fetching
│   └── platform_data_fetcher.py  # Operator/datafield fetching
├── database/                      # Data persistence layer
│   ├── schema.py                 # Database schema and initialization
│   └── operations.py             # Data operations and queries
├── utils/                         # Shared utilities
│   ├── bootstrap.py              # Project path setup
│   ├── helpers.py                # Common helper functions
│   └── cython_helper.py          # Cython compilation utilities
├── config/                        # Configuration management
└── scripts/                       # Standalone utility scripts
```

### File Organization Principles

**Separation of Concerns**: Each directory handles a specific system layer
- `analysis/`: All data processing and visualization logic
- `api/`: External system integration and data fetching
- `database/`: Data persistence and schema management
- `utils/`: Shared utilities and bootstrapping
- `config/`: Configuration and environment management

**Modularity**: Large components are broken into focused sub-modules
- Dashboard is organized into callbacks, components, layouts, and services
- Clustering includes algorithms, feature engineering, and validation
- API layer separates authentication, fetching, and platform data

---

## Core Components

### Database Layer (`database/`)

**Purpose**: Manages all data persistence, schema definition, and database operations.

**Key Files**:
- `schema.py`: Database schema definition, connection pooling, and initialization
- `operations.py`: Alpha data operations with transaction management
- `operations_unsubmitted.py`: Specialized operations for unsubmitted alphas

**Responsibilities**:
- Define and maintain database schema across multiple regions
- Provide connection pooling for high-concurrency access
- Handle batch operations for performance optimization
- Manage database migrations and schema updates

### API Layer (`api/`)

**Purpose**: Handles integration with external WorldQuant Brain API and data fetching.

**Key Files**:
- `auth.py`: Complex authentication flow with session management and biometric support
- `alpha_fetcher.py`: Multi-threaded alpha data retrieval with retry logic
- `platform_data_fetcher.py`: Operator and datafield fetching based on user tier

**Responsibilities**:
- Authenticate and maintain sessions with WorldQuant Brain
- Fetch alpha metadata, PNL data, and platform definitions
- Handle rate limiting, retry logic, and error recovery
- Cache session data for performance optimization

### Analysis Layer (`analysis/`)

**Purpose**: Core business logic for data processing, correlation analysis, and clustering.

**Key Files**:
- `alpha_expression_parser.py`: Parses alpha expressions to extract operators and datafields
- `correlation/correlation_engine.py`: High-performance correlation calculations using Cython
- `clustering/clustering_algorithms.py`: Advanced clustering with UMAP and HDBSCAN
- `analysis_operations.py`: Orchestrates analysis workflows

**Responsibilities**:
- Parse complex alpha expressions to understand composition
- Calculate correlations between alpha strategies using optimized algorithms
- Perform clustering analysis to identify similar alphas
- Generate analysis reports and summaries

### Dashboard Architecture (`analysis/dashboard/`)

**Purpose**: Interactive web interface for data exploration and visualization.

**Component Organization**:
- `app.py`: Application factory and configuration
- `callbacks/`: Event handlers for user interactions (10+ specialized modules)
- `components/`: Reusable UI components for filters, charts, and tables
- `layouts/`: Page layouts and component composition
- `services/`: Business logic separation for data access and processing

**Responsibilities**:
- Provide interactive visualizations of alpha data and relationships
- Handle user filtering and data exploration workflows
- Generate dynamic charts and correlation matrices
- Manage application state and user interactions

### Utility Modules (`utils/`)

**Purpose**: Shared utilities and bootstrapping functionality.

**Key Files**:
- `bootstrap.py`: Automatic project path detection and configuration
- `cython_helper.py`: Automatic Cython compilation and status checking
- `helpers.py`: Common functions for logging, validation, and reporting
- `clustering_utils.py`: Specialized utilities for clustering operations

**Responsibilities**:
- Provide consistent project setup across all scripts
- Handle Cython compilation automatically
- Offer common utilities for logging and data validation
- Support clustering and analysis operations

---

## Data Flow & Processing

### Data Ingestion Pipeline

1. **Authentication**: Establish secure session with WorldQuant Brain API
2. **Data Fetching**: Retrieve alpha metadata, PNL data, and platform definitions
3. **Data Validation**: Validate data integrity and handle missing values
4. **Database Storage**: Store data using optimized batch operations
5. **Cache Management**: Update local caches for performance

### Analysis Pipeline

1. **Expression Parsing**: Extract operators and datafields from alpha expressions
2. **Correlation Calculation**: Compute pairwise correlations using Cython optimization
3. **Clustering Analysis**: Perform dimensionality reduction and clustering
4. **Summary Generation**: Create analysis summaries and statistics
5. **Visualization Preparation**: Format data for dashboard consumption

---

## Database Design

### Schema Architecture

**Core Entities**:
- `regions`: Geographic regions (USA, EUR, CHN, etc.)
- `alphas`: Alpha metadata and performance metrics
- `pnl_*`: Regional PNL data tables (partitioned by region)
- `correlation_*`: Regional correlation statistics
- `alpha_analysis_cache`: Parsed expression data
- `datafields`: Platform operator and datafield definitions

**Design Principles**:
- **Regional Partitioning**: Separate tables per region for performance and maintainability
- **JSONB Usage**: Flexible storage for operators and datafields using PostgreSQL's JSONB
- **Indexing Strategy**: Comprehensive indexes for common query patterns
- **Referential Integrity**: Foreign keys and constraints ensure data consistency

### Performance Optimizations

**Connection Pooling**: SQLAlchemy engine with 50 base connections and 100 overflow capacity
**Batch Operations**: PostgreSQL COPY for high-throughput data insertion
**Query Optimization**: Composite indexes for multi-column filtering
**Partitioning**: Region-based table partitioning for improved query performance

---

## Performance Architecture

### Cython Optimization

**Critical Path Optimization**: Correlation calculations use compiled Cython code for large speedup
**Automatic Compilation**: System automatically detects and compiles Cython modules on startup
**Platform Support**: Cross-platform compilation for Windows, Linux, and macOS

### Caching Strategy

**Two-Tier Caching System**:
- **File System Cache**: API data stored as JSON/CSV files (`data/operators_dynamic.json`, `data/datafields_dynamic.csv`)
- **Database Cache**: Analysis results cached in PostgreSQL table (`alpha_analysis_cache`)

### Concurrent Processing

**Multi-threading**: ThreadPoolExecutor for parallel API calls and data processing
**Connection Pooling**: High-concurrency database access with connection health checks
**Async Support**: Foundation for future async/await implementation

---

## Security & Authentication

### WorldQuant Brain Integration

**Session Management**: Persistent session handling with automatic reauthentication
**Biometric Support**: Interactive biometric authentication flow when required
**Credential Security**: Secure storage of credentials with local file encryption
**Rate Limiting**: Client-side rate limiting to respect API constraints

---

## Deployment Architecture

### Production Deployment

**WSGI Compatibility**: Standard WSGI interface for production deployment
**Process Management**: Support for multiple worker processes with Gunicorn/Waitress
---

This technical documentation provides a comprehensive understanding of the Alpha DataBank system's architecture, components, and design principles. For detailed command-line usage and code examples, refer to [SCRIPTS_AND_COMMANDS.md](SCRIPTS_AND_COMMANDS.md).