# Database Initialization Guide

## ✅ **Updated Scripts (Fixed)**

All main scripts now **automatically create the missing columns** including `excluded` and `exclusion_reason`. No manual migration needed for new installations!

## 🚀 **For New Users (First Time Setup)**

```bash
# Option 1: Initialize database only
python scripts/init_database.py

# Option 2: Initialize + fetch data
python scripts/run_alpha_databank.py --all

# Option 3: Start dashboard (auto-initializes if needed)
python run_analysis_dashboard.py
```

## 🔄 **For Updating Operator/Datafield Analysis**

```bash
# Fetch fresh operators/datafields and recalculate all exclusions
python run_analysis_dashboard.py --renew

# Clear analysis cache only (force re-analysis with existing data)
python run_analysis_dashboard.py --clear-cache
```

All of these will now create:
- ✅ Main database tables (`alphas`, `regions`, `pnl_*`)
- ✅ Analysis tables (`alpha_analysis_cache`, `analysis_summary`) 
- ✅ Unsubmitted alpha tables (`alphas_unsubmitted`, etc.)
- ✅ All required columns including `excluded` and `exclusion_reason`
- ✅ Performance indexes for faster queries

## 🔄 **For Existing Users (Migration)**

If you already have a database but see "column excluded does not exist" errors:

**Quick Fix:**
```bash
psql -d your_database_name -f migrate_analysis_cache.sql
```


## 📋 **What Each Script Does**

| Script | Creates Basic Tables | Creates Analysis Tables | Creates Unsubmitted Tables |
|--------|:-------------------:|:----------------------:|:--------------------------:|
| `scripts/init_database.py` | ✅ | ✅ | ✅ |
| `scripts/run_alpha_databank.py` | ✅ | ✅ | ❌* |
| `run_analysis_dashboard.py` | ❌ | ✅ (auto-check) | ❌ |

*Unsubmitted tables created only when using `--unsubmitted` flag

## 🎯 **Summary**

**New installations**: Just run any main script - everything is set up automatically!  
**Existing installations**: Run the one-time migration, then you're all set!

The "column does not exist" issue is now permanently fixed for all future users. 🎉