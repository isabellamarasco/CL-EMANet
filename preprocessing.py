import pandas as pd
import numpy as np
import os
import argparse
from sklearn.preprocessing import LabelEncoder

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and normalize CICIDS2017 and UNSW-NB15 datasets"
    )
    parser.add_argument(
        "--data_name",
        type=str,
        choices=["CICIDS", "UNSW-NB", "all"],
        required=True,
        help="Dataset to process: CICIDS, UNSW-NB, or all"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["preprocess_only", "normalize_only", "all"],
        required=True,
        help="Operation mode: preprocess_only, normalize_only, all"
    )
    return parser.parse_args()

# ==== CICIDS2017 Functions ====
def merge_cicids_files():
    """Merge multiple CICIDS2017 CSV files into a single dataframe"""
    print("Merging CICIDS_2017 dataset files...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/CIC-IDS-2017", exist_ok=True)

    file_info = [
        ("data/CIC-IDS-2017/Monday-WorkingHours.pcap_ISCX.csv", "Monday", "AllDay"),
        ("data/CIC-IDS-2017/Tuesday-WorkingHours.pcap_ISCX.csv", "Tuesday", "AllDay"),
        ("data/CIC-IDS-2017/Wednesday-workingHours.pcap_ISCX.csv", "Wednesday", "AllDay"),
        ("data/CIC-IDS-2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "Thursday", "Morning"),
        ("data/CIC-IDS-2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "Thursday", "Afternoon"),
        ("data/CIC-IDS-2017/Friday-WorkingHours-Morning.pcap_ISCX.csv", "Friday", "Morning"),
        ("data/CIC-IDS-2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "Friday", "Afternoon"),
        ("data/CIC-IDS-2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "Friday", "Afternoon"),
    ]

    columns_to_drop = [
        " Bwd PSH Flags", " Fwd URG Flags", " Bwd URG Flags", " CWE Flag Count",
        "Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk", " Fwd Avg Bulk Rate",
        " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    ]

    dataframes = []
    for file, day, time in file_info:
        if os.path.exists(file):
            df = pd.read_csv(file)
            df["Day"] = day
            df["Time"] = time
            if " Label" in df.columns:
                df = df.rename(columns={" Label": "Label"})
            df["Label_bin"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
            dataframes.append(df)
            print(f"File {file} merged.")
        else:
            print(f"File {file} not found. Skipped.")

    if not dataframes:
        print("No CICIDS files found.")
        return None

    merged_df = pd.concat(dataframes, ignore_index=True)
    existing_cols = set(merged_df.columns) & set(columns_to_drop)
    merged_df = merged_df.drop(columns=existing_cols)
    return merged_df

def preprocess_cicids(df=None):
    """Preprocess CICIDS dataset without normalization"""
    if df is None:
        if os.path.exists("data/CICIDS_2017.csv"):
            print("Loading merged CICIDS data...")
            df = pd.read_csv("data/CICIDS_2017.csv")
        else:
            print("Merged data not found, merging files...")
            df = merge_cicids_files()
            if df is None:
                print("CICIDS data unavailable.")
                return None
    
    print("Preprocessing CICIDS_2017...")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Remove underrepresented attacks
    Gamma = 50
    counts = df['Label'].value_counts()
    df = df[df['Label'].isin(counts[counts >= Gamma].index)]
    
    return df

def normalize_cicids(df=None):
    """Apply global normalization to preprocessed CICIDS dataset"""
    if df is None:
        if os.path.exists("data/CICIDS_2017_preprocessed.csv"):
            print("Loading preprocessed CICIDS data...")
            df = pd.read_csv("data/CICIDS_2017_preprocessed.csv")
        else:
            print("Preprocessed data not found, running preprocessing...")
            df = preprocess_cicids()
            if df is None:
                print("Failed to preprocess CICIDS data.")
                return None
    
    print("Applying global normalization to CICIDS_2017...")
    
    # Extract non-numeric columns before normalization
    non_numeric_cols = {}
    for col in ['Label', 'Label_bin', 'Time', 'Day']:
        if col in df.columns:
            non_numeric_cols[col] = df[col]
    
    # Normalize numeric columns
    numeric_df = df.select_dtypes(include="number")
    normalized = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())
    
    # Add back non-numeric columns
    for col, values in non_numeric_cols.items():
        normalized[col] = values
    
    return normalized

# ==== UNSW-NB15 Functions ====
def merge_unsw_files():
    """Merge multiple UNSW-NB15 CSV files into a single dataframe"""
    print("Merging UNSW-NB15 files...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/UNSW-NB15", exist_ok=True)
    
    features_file = "data/UNSW-NB15/NUSW-NB15_features.csv"
    if not os.path.exists(features_file):
        print(f"Features file {features_file} not found. Cannot proceed.")
        return None
        
    try:
        features_df = pd.read_csv(features_file, encoding='latin-1')
        column_names = features_df["Name"].str.strip().tolist()
        
        if not column_names:
            print("Column names could not be extracted from features file.")
            return None
        
        for i, col in enumerate(column_names):
            if col == "ct_src_ ltm":
                column_names[i] = "ct_src_ltm"
        
        files = [f"data/UNSW-NB15/UNSW-NB15_{i}.csv" for i in range(1, 5)]
        existing_files = [f for f in files if os.path.exists(f)]
        
        if not existing_files:
            print("No UNSW files found.")
            return None
        
        print(f"Found {len(existing_files)} UNSW data files.")
        
        temp_files = []
        for i, file in enumerate(existing_files):
            try:
                with open(file, 'r', encoding='latin-1', errors='replace') as f:
                    content = f.readlines()
                
                temp_file = f"data/UNSW-NB15/temp_UNSW-NB15_{i+1}.csv"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(",".join(column_names) + "\n")
                    f.writelines(content)
                
                temp_files.append(temp_file)
                print(f"Processed file {file}")
            except Exception as e:
                print(f"Error processing file {file}: {e}")
        
        if not temp_files:
            print("Failed to process any files.")
            return None
        
        dataframes = []
        for temp_file in temp_files:
            try:
                try:
                    df = pd.read_csv(temp_file, encoding='latin-1', on_bad_lines='skip', low_memory=False)
                except TypeError:
                    df = pd.read_csv(temp_file, encoding='latin-1', error_bad_lines=False, low_memory=False)
                dataframes.append(df)
                print(f"Successfully read {temp_file}")
            except Exception as e:
                print(f"Error reading {temp_file}: {e}")
        
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        if not dataframes:
            print("Failed to read any processed files.")
            return None
            
        merged_df = pd.concat(dataframes, ignore_index=True)
        print(f"UNSW files merged successfully with correct headers. Shape: {merged_df.shape}")
        return merged_df
        
    except Exception as e:
        print(f"Error during UNSW merge process: {e}")
        import traceback
        traceback.print_exc()
        return None

def preprocess_unsw(df=None):
    """Preprocess UNSW-NB15 dataset without normalization"""
    if df is None:
        if os.path.exists("data/UNSW-NB15.csv"):
            print("Loading merged UNSW data...")
            df = pd.read_csv("data/UNSW-NB15.csv", low_memory=False)
        else:
            print("Merged data not found, merging files...")
            df = merge_unsw_files()
            if df is None:
                print("UNSW data unavailable.")
                return None
    
    print("Preprocessing UNSW-NB15...")
    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(r'Ã¯Â»Â¿', '', regex=True).str.strip()
    
    df = df.drop(columns=[col for col in df.columns if df[col].isna().any()], errors='ignore')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    for ip_col in ['srcip', 'dstip']:
        if ip_col in df.columns:
            try:
                ip_split = df[ip_col].str.split('.', expand=True)
                for col_idx in ip_split.columns:
                    ip_split[col_idx] = pd.to_numeric(ip_split[col_idx], errors='coerce')
                
                ip_split.columns = [f"{ip_col}_{i+1}" for i in range(len(ip_split.columns))]
                df = pd.concat([df.drop(columns=[ip_col]), ip_split], axis=1)
            except Exception as e:
                print(f"Error processing IP column {ip_col}: {e}")
                df = df.drop(columns=[ip_col])

    encoder = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = encoder.fit_transform(df[col])
        except Exception as e:
            print(f"Error encoding column {col}: {e}")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(-1).astype(int)
    
    return df

def normalize_unsw(df=None):
    """Apply global normalization to preprocessed UNSW-NB15 dataset"""
    if df is None:
        if os.path.exists("data/UNSW-NB15.csv"):
            print("Loading preprocessed UNSW data...")
            df = pd.read_csv("data/UNSW-NB15.csv")
        else:
            print("Preprocessed data not found, running preprocessing...")
            df = preprocess_unsw()
            if df is None:
                print("Failed to preprocess UNSW data.")
                return None
    
    print("Applying global normalization to UNSW-NB15...")
    
    # Identify columns to exclude from normalization
    exclude = [col for col in ['attack_cat', 'Label'] if col in df.columns]
    
    # Store excluded columns
    excluded_values = {col: df[col] for col in exclude}
    
    # Normalize remaining columns
    df_to_normalize = df.drop(columns=exclude)
    df_norm = (df_to_normalize - df_to_normalize.min()) / (df_to_normalize.max() - df_to_normalize.min())
    
    # Add back excluded columns
    for col, values in excluded_values.items():
        df_norm[col] = values
    
    return df_norm

def process_dataset(data_name, mode):
    """Process the specified dataset with the given mode"""
    print(f"\n=== Processing {data_name} with mode: {mode} ===\n")
    
    dataset_config = {
        "CICIDS": {
            "merge_func": merge_cicids_files,
            "preprocess_func": preprocess_cicids,
            "normalize_func": normalize_cicids,
            "base_path": "data/CICIDS_2017",
            "name": "CICIDS_2017"
        },
        "UNSW-NB": {
            "merge_func": merge_unsw_files,
            "preprocess_func": preprocess_unsw,
            "normalize_func": normalize_unsw,
            "base_path": "data/UNSW-NB15",
            "name": "UNSW-NB15"
        }
    }

    if data_name not in dataset_config:
        print(f"Unknown dataset: {data_name}")
        return
    
    config = dataset_config[data_name]
    
    if mode == "preprocess_only":
        output_path = f"{config['base_path']}.csv"
        
        df = config["preprocess_func"]()
        if df is not None:
            df.to_csv(output_path, index=False)
            print(f"Preprocessed {config['name']} saved to '{output_path}'")
    
    elif mode == "normalize_only":
        output_path = f"data/normalized{config['name']}.csv"
        
        df = config["normalize_func"]()
        if df is not None:
            df.to_csv(output_path, index=False)
            print(f"Normalized {config['name']} saved to '{output_path}'")
    
    elif mode == "all":
        preprocessed_path = f"{config['base_path']}.csv"
        normalized_path = f"data/normalized{config['name']}.csv"
        
        df_merged = config["merge_func"]()
        df_preprocessed = config["preprocess_func"](df_merged)
        if df_preprocessed is not None:
            df_preprocessed.to_csv(preprocessed_path, index=False)
            print(f"Preprocessed {config['name']} saved to '{preprocessed_path}'")
            
            df_normalized = config["normalize_func"](df_preprocessed)
            if df_normalized is not None:
                df_normalized.to_csv(normalized_path, index=False)
                print(f"Normalized {config['name']} saved to '{normalized_path}'")

def main():
    """Main entry point for the script"""
    args = parse_args()

    if args.data_name == "all":
        datasets = ["CICIDS", "UNSW-NB"]
        for dataset in datasets:
            process_dataset(dataset, args.mode)
    else:
        process_dataset(args.data_name, args.mode)

if __name__ == "__main__":
    main()