# Alpha Expression Analysis Dashboard

## Overview

This enhanced dashboard provides comprehensive analysis of your alpha expressions, tracking operator and datafield usage patterns across your alpha strategies. It integrates seamlessly with the existing clustering visualization system.

## Features

### 🔍 Expression Analysis
- **Operator Analysis**: Track usage of all 180+ operators with unique and nominal counting
- **Datafield Analysis**: Monitor usage of 1000+ datafields grouped by categories
- **Cross Analysis**: Analyze correlations between operator and datafield usage patterns

### 🎯 Interactive Filtering
- Filter by **Region** (EUR, USA, CHN, etc.)
- Filter by **Universe** (TOP500, TOP1000, TOP2000, etc.)
- Filter by **Delay** (0, 1, 2, etc.)
- Filter by **Alpha Type** (REGULAR, SUPER, UNSUBMITTED)

### 📊 Beautiful Visualizations
- **Bar Charts**: Top 20 most used operators/datafields
- **Pie Charts**: Datafield usage by category
- **Sunburst Charts**: Hierarchical operator usage (coming soon)
- **Heatmaps**: Co-occurrence patterns (coming soon)

### ⚡ Performance Features
- **Smart Caching**: Hybrid precalculation + on-the-fly analysis
- **Incremental Updates**: Automatic cache refresh for new alphas
- **Background Processing**: Non-blocking analysis operations

## Quick Start

### 1. Initialize Database
```bash
# Initialize the analysis schema
python run_analysis_dashboard.py --init-db
```

### 2. Run Dashboard
```bash
# Start the dashboard (opens browser automatically)
python run_analysis_dashboard.py

# Or with custom port
python run_analysis_dashboard.py --port 8051

# Or without browser auto-launch
python run_analysis_dashboard.py --no-browser
```

### 3. Access Dashboard
Open your browser to: `http://localhost:8050`

## Dashboard Tabs

### 📈 Expression Analysis Tab
The main analysis interface with three sub-tabs:

#### ⚙️ Operators Sub-tab
- View top 20 most used operators
- See unique vs nominal usage statistics  
- Analyze operator distribution patterns

#### 📊 Datafields Sub-tab  
- View top 20 most used datafields
- See datafield usage by category (fundamental, technical, analyst, etc.)
- Analyze data source preferences

#### 🔄 Cross Analysis Sub-tab
- Operator-datafield co-occurrence analysis
- Alpha complexity metrics
- Strategy pattern identification

### 🎯 Alpha Clustering Tab
The existing clustering visualization (if clustering data is available):
- MDS, t-SNE, UMAP, PCA visualizations
- Interactive alpha exploration
- WorldQuant Brain integration

## Technical Details

### Architecture
- **Parser**: `analysis/alpha_expression_parser.py` - Extracts operators/datafields from expressions
- **Operations**: `analysis/analysis_operations.py` - Handles database operations and caching
- **Server**: `analysis/clustering/visualization_server.py` - Enhanced web dashboard
- **Database**: PostgreSQL with JSONB storage for flexible analysis data

### Database Schema
- `alpha_analysis_cache` - Stores parsed expression data per alpha
- `analysis_summary` - Aggregated results for fast queries
- Indices optimized for filtering by region, universe, delay, alpha_type

### Counting Methods
- **Unique Count**: Each operator/datafield counted once per alpha (e.g., rank used 3x = 1)
- **Nominal Count**: Total occurrences across all alphas (e.g., rank used 3x = 3)

## Usage Examples

### Basic Analysis
1. Open dashboard
2. Go to "Expression Analysis" tab
3. Click "Apply Filters" to load all data
4. Explore operators and datafields tabs

### Filtered Analysis
1. Select specific region (e.g., "USA")
2. Select universe (e.g., "TOP1000") 
3. Click "Apply Filters"
4. View filtered results

### Integration with Clustering
1. Generate clustering data first
2. Run: `python run_analysis_dashboard.py --data-file clustering_results.json`
3. Access both analysis and clustering tabs

## Performance Notes

- **First Load**: May take 10-30 seconds to parse all alpha expressions
- **Subsequent Loads**: Uses cached data for instant results
- **Memory Usage**: ~50-200MB depending on number of alphas
- **Recommended**: 8GB+ RAM for large datasets (10,000+ alphas)

## Troubleshooting

### Common Issues

**Dashboard won't start:**
- Check if port 8050 is available
- Ensure database connection is working
- Verify operators.txt and datafields CSV exist

**Analysis tab shows "unavailable":**
- Run with `--init-db` flag first  
- Check file paths for operators.txt and all_datafields_comprehensive.csv
- Verify database schema is initialized

**Slow performance:**
- Clear analysis cache: Delete from `alpha_analysis_cache` table
- Restart dashboard to rebuild cache
- Consider filtering to smaller datasets

### Dependencies
- dash >= 2.0.0
- dash-bootstrap-components >= 1.0.0
- plotly >= 5.0.0
- pandas >= 1.3.0
- numpy >= 1.20.0
- sqlalchemy >= 1.4.0
- psycopg2 >= 2.8.0

## Advanced Features

### API Integration
The analysis system can be extended to work with WorldQuant Brain API for real-time alpha data.

### Custom Analysis
Add custom analysis functions to `analysis_operations.py` for specific research needs.

### Export Features
Results can be exported to CSV/JSON for further analysis in external tools.

---

**🚀 Ready to analyze your alpha strategies? Run `python run_analysis_dashboard.py` to get started!**