# dataset_info.py
from pathlib import Path
import yaml

def analyze_datasets(data_path):
    """
    Analyze original dataset and client splits
    Args:
        data_path (str): Path to dataset directory
    """
    data_path = Path(data_path)
    stats = {}
    
    # Read yaml file for class names
    with open(data_path / 'data.yaml', 'r') as f:
        yaml_data = yaml.safe_load(f)
        class_names = yaml_data['names']
    
    # Analyze original dataset
    stats['original'] = {
        'train': count_files(data_path / 'train'),
        'valid': count_files(data_path / 'valid'),
        'test': count_files(data_path / 'test')
    }
    
    # Analyze client splits
    client_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('client_')]
    for client_dir in client_dirs:
        stats[client_dir.name] = {
            'train': count_files(client_dir / 'train'),
            'valid': count_files(client_dir / 'valid'),
            'test': count_files(client_dir / 'test')
        }
    
    # Generate report
    print("\n=== Dataset Analysis Report ===")
    print("\nOriginal Dataset:")
    print_stats(stats['original'])
    
    print("\nClient Splits:")
    for client in [k for k in stats.keys() if k != 'original']:
        print(f"\n{client.upper()}:")
        print_stats(stats[client])
        
    return stats

def count_files(path):
    """Count files and analyze class distribution in labels"""
    img_count = len(list((path / 'images').glob('*')))
    label_count = len(list((path / 'labels').glob('*')))
    
    return {
        'images': img_count,
        'labels': label_count
    }

def print_stats(stats):
    """Pretty print statistics"""
    for split, counts in stats.items():
        print(f"{split}:")
        print(f"  Images: {counts['images']}")
        print(f"  Labels: {counts['labels']}")

if __name__ == "__main__":
    # Example usage
    analyze_datasets("/home/localssk23/Downloads/YOLO")