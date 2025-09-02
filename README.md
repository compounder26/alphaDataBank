# Alpha DataBank

A PostgreSQL-based system for tracking and analyzing WorldQuant Brain alpha trading strategies.

## Setup

### 1. Install Requirements

Install Requirements with a Virtual Environment

It's a must to create a virtual environment (venv) to be able to run this code base, because some scripts of this repo get its dependency from the venv dir (if you don't create venv it won't run properly!).

Create the virtual environment:
```bash
python -m venv venv
```

Activate the virtual environment:

**On Windows:**
```bash
venv\Scripts\activate

**On macOS/Linux:**
```bash

Install the required packages:
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
{
    "email": "email_here", 
    "password": "password_here"
}
```

**.env**

In the root folder create .env file with below structure:
```env
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

> **Note:** Right now you might need to run this script twice, there is a minor bug where on the first try after you authenticate it showed failed but the session cookies is actually stored and it will run successfully on the second try.

### Run Analytics Dashboard
```bash
python run_analysis_dashboard.py
```

### Fetch Unsubmitted Alphas
```bash
python scripts/run_alpha_databank.py --unsubmitted --url "https://api.worldquantbrain.com/users/self/alphas?limit=50&offset=9800&status=UNSUBMITTED%1FIS_FAIL&order=-dateCreated&hidden=false" --all
```
