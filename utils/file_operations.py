"""File system utilities for directory management."""

import shutil
from pathlib import Path


def clear_directory(directory_path: str | Path) -> None:
    """Ensure a directory exists and is empty.
    
    Args:
        directory_path (str | Path): Path to the directory to manage.
    
    Note:
        Creates directory (and parents) if it doesn't exist.
        Removes all contents if directory exists.
        Handles both files and subdirectories.
    """
    directory_path = Path(directory_path)  # Convert to Path object if it's a string
    
    # Create directory if it doesn't exist
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory {directory_path}")
        return  # No need to clear a newly created directory
    
    # Clear existing directory
    for item in directory_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    print(f"Cleared contents of {directory_path}")