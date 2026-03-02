import csv
import os

def log_result(file_path: str, episode: int, seed: int, mode: str, y_true: str, action: str):
    """
    Appends a single experiment result to a CSV file.
    Creates the file and header if it doesn't exist.
    """
    # 1. Determine if we need to write a header (file is new or empty)
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0
    
    # 2. Calculate if the action was correct (0 or 1)
    correct = 1 if y_true == action else 0
    
    # 3. Open in 'append' mode ('a')
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header only once
        if not file_exists:
            writer.writerow(["episode", "seed", "mode", "y_true", "action", "correct"])
            
        # Write the data row
        writer.writerow([episode, seed, mode, y_true, action, correct])