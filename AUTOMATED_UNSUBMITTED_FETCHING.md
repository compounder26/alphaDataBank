# Automated Unsubmitted Alpha Fetching

This document explains the new automated unsubmitted alpha fetching feature that overcomes the 10,000 alpha API limit by intelligently looping through date ranges and offsets.

## Overview

The existing `--unsubmitted` mode required manually providing a URL with specific date ranges and could only fetch up to 10,000 alphas per URL. The new `--unsubmitted-auto` mode automatically:

1. Loops through sharpe thresholds (e.g., >= 1.0 and <= -1.0)
2. For each threshold, loops through date windows from current date back to 2020
3. For each date window, loops through offsets until no more results
4. Automatically adjusts date windows based on the last fetched alpha's date

## Command Usage

### Basic Usage

```bash
# Fetch ALL unsubmitted alphas with default sharpe thresholds (>= 1.0 and <= -1.0) for all regions
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --all

# Fetch ALL unsubmitted alphas for a specific region
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --region USA
```

### Custom Sharpe Thresholds

```bash
# Fetch only alphas with sharpe >= 2.0
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --sharpe-thresholds "2" --region USA

# Fetch only alphas with sharpe <= -2.0
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --sharpe-thresholds "-2" --region USA

# Fetch alphas with sharpe >= 1.5 AND <= -1.5
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --sharpe-thresholds "1.5,-1.5" --region USA

# Fetch alphas with multiple custom thresholds
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --sharpe-thresholds "2,1.5,-1.5,-2" --all
```

### Advanced Options

```bash
# Use larger batch size for faster fetching (default is 50)
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --batch-size 100 --region USA

# Skip specific operations
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --region USA --skip-pnl-fetch --skip-correlation

# Skip database initialization if already done
./venv/Scripts/python.exe scripts/run_alpha_databank.py --unsubmitted-auto --region USA --skip-init
```

## Command Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--unsubmitted-auto` | Enable automated unsubmitted alpha fetching | Required flag |
| `--sharpe-thresholds` | Comma-separated sharpe thresholds | `"1,-1"` or `"2"` or `"1.5,-1.5,-2"` |
| `--batch-size` | Number of alphas per API request (default: 50) | `--batch-size 100` |
| `--region` | Process specific region | `--region USA` |
| `--all` | Process all configured regions | `--all` |
| `--skip-init` | Skip database initialization | Optional |
| `--skip-alpha-fetch` | Skip alpha metadata fetching | Optional |
| `--skip-pnl-fetch` | Skip PNL data fetching | Optional |
| `--skip-correlation` | Skip correlation calculation | Optional |

## How It Works

### Date Window Algorithm

1. **Start**: Current date
2. **End**: 2020-01-01 (platform launch safety margin)
3. **Process**: For each sharpe threshold:
   - Start with full date range
   - Fetch batches using offset pagination
   - When last alpha's date is older than window end date, adjust window
   - Continue until reaching 2020-01-01

### Sharpe Threshold Processing

- **Positive values**: Treated as `>=` (e.g., `1` becomes `is.sharpe>=1`)
- **Negative values**: Treated as `<=` (e.g., `-1` becomes `is.sharpe<=-1`)
- **Default**: `[1.0, -1.0]` (fetch both high positive and high negative sharpe alphas)

### URL Construction

The system automatically builds URLs like:
```
https://api.worldquantbrain.com/users/self/alphas?limit=50&offset=0&status=UNSUBMITTED%1FIS_FAIL&is.sharpe>=1&dateCreated>=2020-01-01T00:00:00-04:00&dateCreated<2025-08-25T00:00:00-04:00&order=-dateCreated&hidden=false
```

## Comparison with Manual Mode

| Feature | Manual (`--unsubmitted --url`) | Automated (`--unsubmitted-auto`) |
|---------|--------------------------------|----------------------------------|
| **Alpha Limit** | 10,000 per URL | Unlimited (fetches ALL) |
| **User Input** | Requires manual URL construction | Automatic URL generation |
| **Date Ranges** | Fixed, user-specified | Dynamic, automatic adjustment |
| **Sharpe Filtering** | Manual URL encoding | Simple parameter format |
| **Resumability** | Manual URL adjustment needed | Automatic window adjustment |
| **Use Case** | Quick targeted fetches | Complete data collection |

## Error Handling

- **API Failures**: Automatic retry with exponential backoff
- **Rate Limiting**: Respects 429 responses and retry-after headers
- **Authentication**: Validates session before each batch
- **Network Issues**: Connection timeout and retry logic
- **Data Validation**: Skips malformed alpha records

## Performance Considerations

- **Batch Size**: Larger batches (up to 100) may be faster but use more memory
- **Concurrency**: Uses the same parallel PNL fetching as regular alphas
- **API Limits**: Includes delays between requests to be respectful
- **Memory Usage**: Processes alphas in batches to avoid memory issues

## Logging

The system provides detailed logging:

```
2025-08-24 20:02:26,674 - root - INFO - Starting automated unsubmitted alphas processing...
2025-08-24 20:02:26,674 - root - INFO - Using custom sharpe thresholds: [2.0]
...
2025-08-24 20:02:26,827 - root - INFO - --- Processing region: USA ---
...
2025-08-24 20:02:26,828 - root - INFO - Automated unsubmitted alphas processing complete.
```

## Integration with Existing Workflow

The automated fetching integrates seamlessly with the existing pipeline:

1. **Database**: Uses same unsubmitted alphas tables
2. **PNL Fetching**: Same parallel PNL retrieval system  
3. **Correlations**: Same correlation calculation with submitted alphas
4. **Regional Processing**: Same region-based organization

## Validation and Safety

- **Conflicting Modes**: Cannot use `--unsubmitted` and `--unsubmitted-auto` together
- **Parameter Validation**: Validates sharpe threshold format and numeric values
- **Region Validation**: Ensures valid region codes
- **URL Conflicts**: Prevents using `--url` with automated mode
- **Loop Prevention**: Maximum iteration safety limits prevent infinite loops

## Examples of Generated URLs

For sharpe >= 1:
```
https://api.worldquantbrain.com/users/self/alphas?limit=50&offset=0&status=UNSUBMITTED%1FIS_FAIL&dateCreated>=2020-01-01T00:00:00-04:00&dateCreated<2025-08-24T23:59:59-04:00&order=-dateCreated&hidden=false&is.sharpe>=1.0
```

For sharpe <= -1:
```
https://api.worldquantbrain.com/users/self/alphas?limit=50&offset=0&status=UNSUBMITTED%1FIS_FAIL&dateCreated>=2020-01-01T00:00:00-04:00&dateCreated<2025-08-24T23:59:59-04:00&order=-dateCreated&hidden=false&is.sharpe<=-1.0
```