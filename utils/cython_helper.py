"""
Cython Build Helper

Automatically compiles Cython extensions when needed, eliminating the need
for developers to manually run setup.py build_ext --inplace.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path


def ensure_cython_compiled(module_name: str = 'correlation_utils', force_rebuild: bool = False) -> bool:
    """
    Ensure a Cython module is compiled and available for import.

    This function checks if a Cython module is already compiled. If not,
    it automatically compiles it using the setup.py file.

    Args:
        module_name: Name of the Cython module to check/compile (default: correlation_utils)
        force_rebuild: If True, rebuild even if module exists

    Returns:
        bool: True if module is available (either already existed or successfully compiled)
    """
    # Check if module is already available
    if not force_rebuild:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            return True

    # Find project root (where setup.py should be)
    current_dir = Path(__file__).parent.parent  # utils -> project root
    setup_py = current_dir / 'setup.py'
    pyx_file = current_dir / f'{module_name}.pyx'

    # Check if source files exist
    if not setup_py.exists():
        print(f"Warning: setup.py not found at {setup_py}")
        return False

    if not pyx_file.exists():
        print(f"Warning: {module_name}.pyx not found at {pyx_file}")
        return False

    # Attempt to compile
    print(f"Compiling {module_name}.pyx for your system...")
    print("This is a one-time setup that takes about 5-10 seconds.")

    try:
        # Run setup.py build_ext --inplace
        result = subprocess.run(
            [sys.executable, str(setup_py), 'build_ext', '--inplace'],
            cwd=str(current_dir),
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"[OK] Successfully compiled {module_name}")

            # Verify the module can now be imported
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                return True
            else:
                print(f"Warning: Compilation succeeded but {module_name} still cannot be imported")
                print("You may need to restart your Python session")
                return False
        else:
            print(f"[ERROR] Failed to compile {module_name}")
            print(f"Error output: {result.stderr}")

            # Check for common issues
            if "Microsoft Visual C++" in result.stderr or "error: Microsoft Visual" in result.stderr:
                print("\nPossible solution:")
                print("- On Windows: Install Microsoft Visual C++ Build Tools")
                print("  Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")

            elif "Cython" in result.stderr or "No module named" in result.stderr:
                print("\nPossible solution:")
                print("- Install Cython: pip install Cython")

            elif "numpy" in result.stderr:
                print("\nPossible solution:")
                print("- Install NumPy: pip install numpy")

            return False

    except Exception as e:
        print(f"[ERROR] Exception during compilation: {e}")
        return False


def get_correlation_utils():
    """
    Get the correlation_utils module, compiling if necessary.

    This is a convenience function that ensures the module is compiled
    and then imports and returns it.

    Returns:
        module: The imported correlation_utils module, or None if compilation fails
    """
    if ensure_cython_compiled('correlation_utils'):
        try:
            import correlation_utils
            return correlation_utils
        except ImportError as e:
            print(f"Failed to import correlation_utils after compilation: {e}")
            return None
    else:
        print("Warning: Could not compile correlation_utils")
        print("Correlation calculations will fall back to slower Python implementation")
        return None


def check_cython_status():
    """
    Check and report the status of all Cython modules in the project.

    Useful for debugging and setup verification.
    """
    print("Cython Module Status Check")
    print("-" * 50)

    # Check for Cython installation
    try:
        import Cython
        print(f"[OK] Cython installed (version {Cython.__version__})")
    except ImportError:
        print("[ERROR] Cython not installed - run: pip install Cython")
        return

    # Check for NumPy (required for correlation_utils)
    try:
        import numpy as np
        print(f"[OK] NumPy installed (version {np.__version__})")
    except ImportError:
        print("[ERROR] NumPy not installed - run: pip install numpy")
        return

    # Check for compiler on Windows
    if sys.platform == 'win32':
        try:
            result = subprocess.run(['cl'], capture_output=True, text=True)
            if result.returncode == 0 or 'Microsoft' in result.stderr:
                print("[OK] Microsoft Visual C++ compiler found")
        except FileNotFoundError:
            print("[WARNING] Microsoft Visual C++ compiler not found")
            print("  Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")

    # Check correlation_utils module
    try:
        import correlation_utils
        print("[OK] correlation_utils module is compiled and ready")

        # Check for the compiled file
        import glob
        current_dir = Path(__file__).parent.parent
        compiled_files = glob.glob(str(current_dir / "correlation_utils*.pyd")) + \
                        glob.glob(str(current_dir / "correlation_utils*.so"))
        if compiled_files:
            for f in compiled_files:
                size_kb = os.path.getsize(f) / 1024
                print(f"  Found: {os.path.basename(f)} ({size_kb:.1f} KB)")
    except ImportError:
        print("[ERROR] correlation_utils module not compiled")
        print("  Run: python setup.py build_ext --inplace")
        print("  Or it will be auto-compiled on first use")

    print("-" * 50)


if __name__ == "__main__":
    # If run directly, perform status check
    check_cython_status()