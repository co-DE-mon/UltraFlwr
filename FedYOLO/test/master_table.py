from FedYOLO.config import SERVER_CONFIG
from FedYOLO.config import DATASET_NAME, HOME, DATASET_PATH, NUM_CLIENTS
import yaml
import pandas as pd
import numpy as np

DATASET = DATASET_NAME
CONFIG_STRATEGY = SERVER_CONFIG['strategy']
DATA_YAML = f"{DATASET_PATH}/data.yaml"

# Get class names from data.yaml
with open(DATA_YAML) as file:
    class_names = yaml.load(file, Loader=yaml.FullLoader)['names']

print(f"Dataset: {DATASET} | Strategy: {CONFIG_STRATEGY}")

def should_skip_server():
    """Check if current strategy should skip server-based results."""
    return any(keyword in CONFIG_STRATEGY.lower() for keyword in ['head', 'neck', 'backbone'])

def generate_result_paths(num_clients):
    """Generate paths for all client and server results."""
    paths = {}
    
    # Client results (both client-side and server-side evaluations)
    for client_id in range(num_clients):
        paths[f"client_{client_id}_results_client"] = (
            f"{HOME}/results/client_{client_id}_results_{DATASET}_{CONFIG_STRATEGY}.csv"
        )
        paths[f"client_{client_id}_results_server"] = (
            f"{HOME}/results/client_{client_id}_results_{DATASET}_{CONFIG_STRATEGY}_server.csv"
        )
    
    # Only add server paths if we're not skipping server-based results
    if not should_skip_server():
        # Server results for each client and server itself
        for client_id in range(num_clients):
            paths[f"server_results_client_{client_id}"] = (
                f"{HOME}/results/server_client_{client_id}_results_{DATASET}_{CONFIG_STRATEGY}.csv"
            )
        
        # Server's own results
        paths["server_results_server"] = f"{HOME}/results/server_results_{DATASET}_{CONFIG_STRATEGY}.csv"
    
    return paths

def generate_source_labels(num_clients):
    """Generate source labels for all clients and server combinations."""
    labels = {}
    
    # Client labels
    for client_id in range(num_clients):
        labels[f"client_{client_id}_results_client"] = f'Client {client_id} - Client'
        labels[f"client_{client_id}_results_server"] = f'Client {client_id} - Server'
    
    # Only add server labels if we're not skipping server-based results
    if not should_skip_server():
        for client_id in range(num_clients):
            labels[f"server_results_client_{client_id}"] = f'Server - Client {client_id}'
        labels["server_results_server"] = 'Server - Server'
    
    return labels

def process_results(num_clients):
    """Process results for all clients and create analysis tables."""
    # Generate paths and labels
    result_paths = generate_result_paths(num_clients)
    source_labels = generate_source_labels(num_clients)
    
    # Load CSVs into a dictionary of DataFrames
    dfs = {}
    for key, path in result_paths.items():
        try:
            dfs[key] = pd.read_csv(path)
        except FileNotFoundError:
            print(f"Warning: Could not find results file: {path}")
            continue
    
    if not dfs:
        raise ValueError("No result files were found or could be loaded.")
    
    # Process each DataFrame
    for key, df in dfs.items():
        df['source'] = source_labels[key]
        if key.startswith('server') and not should_skip_server():
            rows_per_class = len(df) // len(class_names)
            df['class'] = np.repeat(class_names, rows_per_class)
    
    # Concatenate all DataFrames into the master table
    master_table = pd.concat(list(dfs.values()), ignore_index=True)
    
    # Pivot the table
    pivoted_table = master_table.pivot_table(
        index='class', 
        columns='source', 
        values=['precision', 'recall', 'mAP50', 'mAP50-95'], 
        aggfunc='first'
    )
    
    # Flatten column multi-index
    pivoted_table.columns = [f'{metric} - {source}' for metric, source in pivoted_table.columns]
    
    # Reshape for better visualization
    pivoted_table = pivoted_table.reset_index()
    pivoted_table = pd.melt(pivoted_table, id_vars='class', var_name='metric - source', value_name='value')
    
    # Create separate tables for each metric
    metric_tables = {
        'precision': pivoted_table[pivoted_table['metric - source'].str.startswith('precision')],
        'recall': pivoted_table[pivoted_table['metric - source'].str.startswith('recall')],
        'mAP50': pivoted_table[pivoted_table['metric - source'].str.startswith('mAP50 -')],
        'mAP50-95': pivoted_table[pivoted_table['metric - source'].str.startswith('mAP50-95')]
    }
    
    return metric_tables

def print_results(metric_tables):
    """Print results for all metrics."""
    if should_skip_server():
        print("\nNote: Server-based results are skipped for head/neck/backbone strategies.")
    
    for metric, table in metric_tables.items():
        print(f"\n{metric} Table:")
        print(table)

# Usage example:
num_clients = NUM_CLIENTS
metric_tables = process_results(num_clients)
print_results(metric_tables)