"""
Progress bar utilities for clean console output.
"""
from tqdm import tqdm
from typing import Optional, Any
import sys
import logging

# Store reference to progress bars to manage them properly
active_bars = {}

def create_progress_bar(total: int, desc: str, position: int = 0, leave: bool = True, unit: str = "items") -> tqdm:
    """
    Create a clean progress bar with minimal output.

    Args:
        total: Total number of items
        desc: Description for the progress bar
        position: Position for nested progress bars
        leave: Whether to leave the progress bar after completion
        unit: Unit of items (e.g., "alphas", "regions")

    Returns:
        tqdm progress bar object
    """
    bar = tqdm(
        total=total,
        desc=desc,
        position=position,
        leave=leave,
        unit=unit,
        ncols=100,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        file=sys.stdout,
        disable=False
    )

    # Store reference
    bar_id = f"{desc}_{position}"
    active_bars[bar_id] = bar

    return bar

def update_progress_bar(bar: tqdm, n: int = 1, desc: Optional[str] = None) -> None:
    """
    Update a progress bar.

    Args:
        bar: The progress bar to update
        n: Number of items to advance
        desc: Optional new description
    """
    if desc:
        bar.set_description(desc)
    bar.update(n)

def close_progress_bar(bar: tqdm) -> None:
    """
    Close a progress bar properly.

    Args:
        bar: The progress bar to close
    """
    bar.close()

def close_all_progress_bars() -> None:
    """Close all active progress bars."""
    for bar in active_bars.values():
        if bar:
            bar.close()
    active_bars.clear()

def print_success(message: str) -> None:
    """
    Print a success message with checkmark.

    Args:
        message: Success message to print
    """
    tqdm.write(f"✓ {message}")

def print_error(message: str) -> None:
    """
    Print an error message.

    Args:
        message: Error message to print
    """
    tqdm.write(f"✗ {message}", file=sys.stderr)

def print_info(message: str) -> None:
    """
    Print an info message.

    Args:
        message: Info message to print
    """
    tqdm.write(f"  {message}")

def print_header(title: str) -> None:
    """
    Print a section header.

    Args:
        title: Title for the section
    """
    tqdm.write(f"\n{title}")
    tqdm.write("=" * len(title))

def configure_minimal_logging() -> None:
    """
    Configure logging to be minimal for production use.
    Only show WARNING and ERROR level messages.
    """
    # Set root logger to WARNING
    logging.getLogger().setLevel(logging.WARNING)

    # Specific modules that should be even quieter
    quiet_modules = [
        'api.auth',
        'api.alpha_fetcher',
        'database.schema',
        'database.operations',
        'database.operations_unsubmitted',
        'analysis.correlation.correlation_engine',
        'urllib3',
        'requests'
    ]

    for module in quiet_modules:
        logging.getLogger(module).setLevel(logging.ERROR)

    # Configure format to be minimal
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))

    # Clear existing handlers and add new one
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

def suppress_retry_logs() -> None:
    """
    Suppress retry attempt logs that eventually succeed.
    Only show final failures.
    """
    # Suppress urllib3 retry warnings
    import urllib3
    urllib3.disable_warnings()

    # Set requests logging to only show critical errors
    logging.getLogger('urllib3').setLevel(logging.CRITICAL)
    logging.getLogger('requests').setLevel(logging.CRITICAL)