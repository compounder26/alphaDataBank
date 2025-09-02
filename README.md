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

Create a `.env` file in the root of the project directory. This file will store your database credentials. Copy the following structure into it, replacing the placeholder values with your actual database details:

```env
DB_PASSWORD=your_db_password
DB_USER=your_db_user
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_db_name
```

Create a `secrets/` directory in the project root. Inside this directory, create a file named `platform-brain.json` with your WorldQuant Brain credentials:

```json
{
    "email": "your_email@example.com",
    "password": "your_brain_password"
}
```

### 5. Initialize the Database

Once your configuration is set, you need to create the database tables. Run the following script from the project root:

```bash
python scripts/init_database.py
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
