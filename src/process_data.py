import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_utils import clean_data, align_and_merge

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    
    # Paths
    spot_path = os.path.join(data_dir, 'nifty_spot_5min.csv')
    futures_path = os.path.join(data_dir, 'nifty_futures_5min.csv')
    options_path = os.path.join(data_dir, 'nifty_options_5min.csv')
    
    # Check if files exist
    if not os.path.exists(spot_path):
        print("Data files not found. Please run fetch_data.py first.")
        return

    # Load Data
    print("Loading data...")
    spot_df = pd.read_csv(spot_path)
    futures_df = pd.read_csv(futures_path)
    options_df = pd.read_csv(options_path)
    
    # 1. Cleaning
    report_lines = []
    report_lines.append("Data Cleaning Report")
    report_lines.append("====================")
    
    clean_spot = clean_data(spot_df, "Spot Data")
    report_lines.append(f"Spot Data: {len(spot_df)} -> {len(clean_spot)} rows")
    
    clean_futures = clean_data(futures_df, "Futures Data")
    report_lines.append(f"Futures Data: {len(futures_df)} -> {len(clean_futures)} rows")
    
    clean_options = clean_data(options_df, "Options Data")
    report_lines.append(f"Options Data: {len(options_df)} -> {len(clean_options)} rows")
    
    # Save cleaned
    clean_spot.to_csv(os.path.join(data_dir, 'nifty_spot_5min_clean.csv'), index=False)
    clean_futures.to_csv(os.path.join(data_dir, 'nifty_futures_5min_clean.csv'), index=False)
    clean_options.to_csv(os.path.join(data_dir, 'nifty_options_5min_clean.csv'), index=False)
    
    # Save Report
    with open(os.path.join(data_dir, 'data_cleaning_report.txt'), 'w') as f:
        f.write("\n".join(report_lines))
    print("Saved data cleaning report.")
    
    # 2. Merging
    merged_df = align_and_merge(clean_spot, clean_futures, clean_options)
    
    # Save Merged
    merged_path = os.path.join(data_dir, 'nifty_merged_5min.csv')
    merged_df.to_csv(merged_path, index=False)
    print(f"Saved Merged Data to {merged_path}")

if __name__ == "__main__":
    main()
