from FedYOLO.config import SERVER_CONFIG, DATASET_NAME, HOME, DATASET_PATH
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

# Define result paths
result_paths = {
    "client_0_results_client": f"{HOME}/results/client_0_results_{DATASET}_{CONFIG_STRATEGY}.csv",
    "client_0_results_server": f"{HOME}/results/client_0_results_{DATASET}_{CONFIG_STRATEGY}_server.csv",
    "client_1_results_client": f"{HOME}/results/client_1_results_{DATASET}_{CONFIG_STRATEGY}.csv",
    "client_1_results_server": f"{HOME}/results/client_1_results_{DATASET}_{CONFIG_STRATEGY}_server.csv",
    "server_results_client_0": f"{HOME}/results/server_client_0_results_{DATASET}_{CONFIG_STRATEGY}.csv",
    "server_results_client_1": f"{HOME}/results/server_client_1_results_{DATASET}_{CONFIG_STRATEGY}.csv",
    "server_results_server": f"{HOME}/results/server_results_{DATASET}_{CONFIG_STRATEGY}.csv"
}

# Load CSVs into a dictionary of DataFrames
dfs = {
    key: pd.read_csv(path) for key, path in result_paths.items()
}

# Add 'source' column to each DataFrame
source_labels = {
    "client_0_results_client": 'Client 0 - Client',
    "client_1_results_client": 'Client 1 - Client',
    "client_0_results_server": 'Client 0 - Server',
    "client_1_results_server": 'Client 1 - Server',
    "server_results_client_0": 'Server - Client 0',
    "server_results_client_1": 'Server - Client 1',
    "server_results_server": 'Server - Server'
}

# Process each DataFrame
for key, df in dfs.items():
    df['source'] = source_labels[key]
    if key.startswith('server'):
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

# Separate tables for each metric
precision_table = pivoted_table[pivoted_table['metric - source'].str.startswith('precision')]
recall_table = pivoted_table[pivoted_table['metric - source'].str.startswith('recall')]
map50_table = pivoted_table[pivoted_table['metric - source'].str.startswith('mAP50 -')]  # Changed pattern
map95_table = pivoted_table[pivoted_table['metric - source'].str.startswith('mAP50-95')]

# Print results
print("\nPrecision Table:")
print(precision_table)
print("\nRecall Table:")
print(recall_table)
print("\nmAP50 Table:")
print(map50_table)
print("\nmAP50-95 Table:")
print(map95_table)