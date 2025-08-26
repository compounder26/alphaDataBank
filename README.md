# Alpha DataBank

A PostgreSQL-based system for tracking and analyzing WorldQuant Brain alpha trading strategies.

## Setup

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. PostgreSQL Setup
- Install PostgreSQL
- Run pgAdmin
- Create your database

### 3. Configuration Files

Create `secrets/` directory with:

**secrets/platform-brain.json**
```json
{"email": "email_here", "password": "password_here"}
```

**.env**
In the root folder create .env file with below structure:
```
DB_PASSWORD=db_password
DB_USER=user_here
DB_HOST=localhost
DB_PORT=port_here
DB_NAME=name_here
```

## Usage

### Fetch All Submitted Alphas
```bash
python scripts/run_alpha_databank.py --all
```

### Run Analytics Dashboard
```bash
python run_analysis_dashboard.py
```

### Fetch Unsubmitted Alphas
```bash
python scripts/run_alpha_databank.py --unsubmitted --url "https://api.worldquantbrain.com/users/self/alphas?limit=50&offset=9800&status=UNSUBMITTED%1FIS_FAIL&order=-dateCreated&hidden=false"
```