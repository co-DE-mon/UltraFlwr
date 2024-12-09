import re

def extract_results_path(file_path):
    with open(file_path, "r") as file:
        file_content = file.read()
    lines = file_content.split('\n')
    for line in reversed(lines):
        if line.startswith("Results saved to"):
            match = re.search(r"Results saved to (.+)", line)
            if match:
                # Remove ANSI escape sequences
                clean_path = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', match.group(1))
                return clean_path.strip()
    return None
