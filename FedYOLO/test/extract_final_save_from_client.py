import re

def extract_results_path(log_file_path):
    """
    Extract the last occurrence of the path prefixed by "Results saved to" from a log file.

    Args:
        log_file_path (str): The path to the log file.

    Returns:
        str: The extracted path, or None if not found.
    """
    try:
        with open(log_file_path, 'r') as file:
            log_content = file.read()
        
        # Find all matches for paths prefixed by "Results saved to"
        matches = re.findall(r"Results saved to .*?(/home/.+?)(?:\x1b|$)", log_content)
        
        # Return the last match if it exists
        return matches[-1] if matches else None
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

# Example usage
log_file_path = "/home/yang/Documents/GitHub/FedDet/client_1_log.txt"
path = extract_results_path(log_file_path)
if path:
    print(f"Extracted path: {path}")
else:
    print("No path found in the log file.")
