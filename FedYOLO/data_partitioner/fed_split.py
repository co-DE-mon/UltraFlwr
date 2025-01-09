import yaml
import shutil
from pathlib import Path
from prettytable import PrettyTable
from FedYOLO.config import SPLITS_CONFIG

def count_classes(label_files):
    """Count occurrences of each class in the label files."""
    class_counts = {}
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    return class_counts

def create_class_distribution_table(global_counts, client_counts, split):
    """Create a table comparing global and client-specific class distributions."""
    classes = sorted(set(global_counts.get(split, {}).keys()).union(
        *[client_counts[client].get(split, {}).keys() for client in client_counts]
    ))
    
    table = PrettyTable()
    table.field_names = ["Class", "Global Count"] + [f"{client} Count" for client in client_counts]
    
    for class_id in classes:
        row = [
            class_id, 
            global_counts.get(split, {}).get(class_id, 0), 
            *[client_counts[client].get(split, {}).get(class_id, 0) for client in client_counts]
        ]
        table.add_row(row)
    
    return table

def split_dataset(config):
    """
    Split dataset for federated learning with n clients
    Args:
        config (dict): Configuration dictionary containing:
            - ratio (list): List of ratios for each client
            - dataset (str): Path to dataset directory
            - num_clients (int): Number of clients
    """
    ratios = config['ratio']
    data_path = Path(config['dataset'])
    num_clients = config['num_clients']

    # Validate inputs
    if not isinstance(ratios, list) or len(ratios) != num_clients:
        raise ValueError(f"Ratios list must have length {num_clients}")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Read original yaml file
    with open(data_path / 'data.yaml', 'r') as f:
        data = yaml.safe_load(f)

    # Count global class-wise information
    global_class_counts = {'train': {}, 'valid': {}, 'test': {}, 'total': {}}
    for split in ['train', 'valid', 'test']:
        label_files = list((data_path / split / 'labels').glob('*'))
        split_class_counts = count_classes(label_files)
        for class_id, count in split_class_counts.items():
            global_class_counts[split][class_id] = count
            global_class_counts['total'][class_id] = global_class_counts['total'].get(class_id, 0) + count

    partition_path = data_path / 'partitions'
    
    # Create client directories and yaml files
    for client_id in range(num_clients):
        client_dir = partition_path / f'client_{client_id}'
        for split in ['train', 'valid', 'test']:
            (client_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (client_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Create client yaml
        client_yaml = data.copy()
        client_yaml['train'] = './train/images'
        client_yaml['val'] = './valid/images'
        client_yaml['test'] = './test/images'

        with open(client_dir / 'data.yaml', 'w') as f:
            yaml.dump(client_yaml, f)

    client_class_counts = {f'client_{i}': {'train': {}, 'valid': {}, 'test': {}} 
                          for i in range(num_clients)}

    # Split and copy files
    for split in ['train', 'valid', 'test']:
        images = list((data_path / split / 'images').glob('*'))
        labels = list((data_path / split / 'labels').glob('*'))
        
        start_idx = 0
        remaining = len(images)

        for client_id in range(num_clients):
            # For last client, use all remaining files
            if client_id == num_clients - 1:
                n_files = remaining
            else:
                n_files = int(len(images) * ratios[client_id])
                remaining -= n_files
            
            client_images = images[start_idx:start_idx + n_files]
            client_labels = labels[start_idx:start_idx + n_files]
            
            client_dir = partition_path / f'client_{client_id}'
            for img in client_images:
                shutil.copy2(img, client_dir / split / 'images')
            for lbl in client_labels:
                shutil.copy2(lbl, client_dir / split / 'labels')
            
            start_idx += n_files
            
            # Count class-wise information
            class_counts = count_classes(client_labels)
            client_class_counts[f'client_{client_id}'][split] = class_counts

        # Print the table for this split
        table = create_class_distribution_table(global_class_counts, client_class_counts, split)
        print(f"\nClass distribution for {split} split:")
        print(table)

if __name__ == "__main__":
    split_dataset(SPLITS_CONFIG)