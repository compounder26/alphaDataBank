# Alpha DataBank

A PostgreSQL-based system for tracking and analyzing WorldQuant Brain alpha trading strategies.

## Setup

### 1. Install Requirements

**Recommended: Use a Virtual Environment**

We strongly recommend using a virtual environment to avoid dependency conflicts and ensure compatibility with your system's other Python projects.

Create the virtual environment:
```bash
python -m venv venv
```

Activate the virtual environment:

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Compile Cython Module (Optional, for Performance)

This project uses a Cython extension (`correlation_utils.pyx`) to significantly speed up correlation calculations. To get this performance boost, you need to compile it. If you skip this step, the code will automatically use a slower pure Python version.

**A. Install a C++ Compiler**

You only need to do this once on your system.

*   **On Windows:**
    1.  Download the **Visual Studio Build Tools** from the [official Microsoft site](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
    2.  Run the installer. In the "Workloads" tab, check the box for **"Desktop development with C++"** and click "Install".

*   **On macOS:**
    Open the Terminal and run:
    ```bash
    xcode-select --install
    ```

*   **On Linux (Debian/Ubuntu):**
    Open the Terminal and run:
    ```bash
    sudo apt update && sudo apt install build-essential
    ```

**B. Compile the Module**

Once you have a compiler, run the following command in the project's root directory (with your virtual environment activated):
```bash
python setup.py build_ext --inplace
```

### 3. PostgreSQL Setup

- Install PostgreSQL.
- Open pgAdmin and create a new, empty database for this project.

### 4. Configure Environment Variables

#### Database Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and update with your PostgreSQL credentials:
   ```env
   DB_USER=postgres
   DB_PASSWORD=your_password_here
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=alpha_database
   ```

#### WorldQuant Brain API Credentials

1. Copy the example credentials file from the secrets directory:
   ```bash
   cp secrets/platform-brain.json.example secrets/platform-brain.json
   ```

2. Edit `secrets/platform-brain.json` with your WorldQuant Brain credentials:
   ```json
   {
       "email": "your_email@example.com",
       "password": "your_brain_password"
   }
   ```

**Note:** The `secrets/` directory and `.env` file are gitignored for security. Never commit these files containing actual credentials.

### 5. Initialize the Database

Once your configuration is set, you need to create the database tables. Run the following script from the project root:

```bash
python scripts/init_database.py
```


## Usage

**Run commands in order: fetch data first, then visualize.**

### 1. Fetch All Submitted Alphas
```bash
python scripts/run_alpha_databank.py --all
```

### 2. Run Analytics Dashboard

#### Option 1: Development Mode (Simple, with warning)
```bash
python run_analysis_dashboard.py
```

This is the traditional way to run the dashboard. It's simple and works well for personal use, but you'll see a warning about using the development server.

#### Option 2: Production Mode

**Windows:**
```bash
waitress-serve --host=127.0.0.1 --port=8050 wsgi:server
```

**Linux/Mac:**
```bash
gunicorn -w 4 -b 127.0.0.1:8050 wsgi:server
```

**Production Mode Commands:**
```bash
# Refresh operators/datafields (--renew equivalent)
python renew_genius.py

# Clear cache (--clear-cache equivalent)
python clear_cache.py

# Regenerate clustering data
python refresh_clustering.py              # All regions
python refresh_clustering.py --regions USA EUR CHN  # Specific regions
```

**Note:** The analysis dashboard excludes super alphas and only visualizes or selects regular alphas for analysis.

#### Dynamic Operator & Datafield Filtering
The system now fetches operators and datafields on-premises based on your WorldQuant Brain tier access. Only alphas using operators/datafields available to your current tier are displayed in the dashboard.

**Fetch user-specific operators/datafields:**
```bash
python run_analysis_dashboard.py --renew
```

The `--renew` flag:
- Fetches operators and datafields from the WorldQuant Brain API based on your account tier
- Caches the data locally for performance
- Automatically clears the analysis cache to re-process all alphas with the updated access list
- Ensures the dashboard only shows operators and datafields you actually have access to

**Note:** Alphas containing operators or datafields not available to your tier are automatically excluded from all statistics and visualizations.

### 3. Fetch Unsubmitted Alphas (Optional, link below is an example, use your own links that contain the alphas you want from the brain platform alphas filter page - you can get it by inspecting the network)
```bash
python scripts/run_alpha_databank.py --unsubmitted --url "https://api.worldquantbrain.com/users/self/alphas?limit=50&offset=9800&status=UNSUBMITTED%1FIS_FAIL&order=-dateCreated&hidden=false" --all
```
