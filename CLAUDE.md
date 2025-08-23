# Claude Code Project Instructions

## Search Tool Preferences

**ALWAYS use `rg` (ripgrep) via Bash tool instead of Grep tool for code searches.**

The Grep tool frequently gets cancelled/interrupted in this codebase environment. Use these reliable patterns:

- File search: `rg -l "pattern" /path --type py`
- Content search: `rg "pattern" /path --type py` 
- Context search: `rg -A 5 -B 5 "pattern" /path`

## Project Structure Notes

### Correlation Scripts Analysis
Located in `/scripts/`, there are 5 correlation calculation scripts:

- **calculate_cross_correlation.py** ✅ - ACTIVELY USED (in Claude settings)
- **calculate_self_correlation_standalone.py** ❌ - UNUSED (demo with fake data)
- **calculate_self_correlation_flexible.py** ❌ - UNUSED (handles 2/3 PNL columns)
- **calculate_self_correlation_api.py** ❌ - UNUSED (concurrent API fetching)
- **calculate_self_correlation.py** ❌ - UNUSED (original version)

Only `calculate_cross_correlation.py` is configured for use in `.claude/settings.local.json`.

## Development Environment

- Uses virtual environment at `./venv/`
- Python scripts executed via: `./venv/Scripts/python.exe scripts/script_name.py`
- Has pre-approved bash commands in `.claude/settings.local.json`