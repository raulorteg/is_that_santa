from pathlib import Path

def setup_directories():
    """
    Create all directories the other functions expect to work with.
    Execute by calling python setup_directories.py
    """
    queue_to_make = ["results", "figures", "model_savepoints", "external"]
    [Path(directory).mkdir(parents=True, exist_ok=True) for directory in queue_to_make]

if __name__ == "__main__":
    setup_directories()
    print("Done! Directories are now created.")