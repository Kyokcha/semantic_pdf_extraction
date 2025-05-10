# utils/file_operations.py

import shutil
from pathlib import Path


def clear_directory(directory_path):
    """
    Ensures a directory exists and is empty.
    If the directory exists, its contents will be removed.
    If it doesn't exist, it will be created (including any parent directories).
    
    Args:
        directory_path (str or Path): Path to the directory to be cleared/created
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