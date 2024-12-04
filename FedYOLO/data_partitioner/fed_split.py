import yaml
import shutil
from pathlib import Path
from FedYOLO.config import SPLITS_CONFIG

def split_dataset(ratios, data_path):

    """
    Split dataset for federated learning
    Args:
        num_clients (int): Number of clients
        ratios (list): List of ratios for each client (must sum to 1)
        data_path (str): Path to dataset directory containing data.yaml
    """

    num_clients = len(ratios)

    # Validate inputs
    if not isinstance(ratios, list) or len(ratios) != num_clients:
        raise ValueError(f"Ratios list must have length {num_clients}")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    data_path = Path(data_path)
    
    # Read original yaml file
    with open(data_path / 'data.yaml', 'r') as f:
        data = yaml.safe_load(f)

    partition_path = data_path / 'partitions'
    
    # Create client directories
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

        # client_yaml['train'] = f'../{client_dir}/train/images'
        # client_yaml['val'] = f'../{client_dir}/valid/images'
        # client_yaml['test'] = f'../{client_dir}/test/images'
        
        with open(client_dir / 'data.yaml', 'w') as f:
            yaml.dump(client_yaml, f)

    #? Rounding error handling via remaining files all in final client
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

# Example usage:
split_dataset(SPLITS_CONFIG['ratio'], SPLITS_CONFIG['dataset'])