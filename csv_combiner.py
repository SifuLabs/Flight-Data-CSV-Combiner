import pandas as pd
import os
import glob
from pathlib import Path
import numpy as np
import gc
import psutil
from tqdm import tqdm

def combine_csv_files(folder_path, value_column=None, timestamp_column=None, output_filename='combined_data.csv', 
                     batch_size=100, precision=4, use_compression=True, optimize_data_types=True, 
                     remove_zero_columns=False, downsample_factor=1, auto_detect_format=True):
    """
    Combine multiple CSV files into one, using the timestamp column as the common key
    and extracting specified value columns from each CSV.
    
    Parameters:
    folder_path (str): Path to the folder containing CSV files
    value_column (str): Name of the column to extract values from (if None, will auto-detect)
    timestamp_column (str): Name of the timestamp column (if None, will auto-detect)
    output_filename (str): Name of the output CSV file
    batch_size (int): Number of files to process in each batch (default: 100)
    precision (int): Number of decimal places to round ONLY numeric values (default: 4)
    use_compression (bool): Whether to use gzip compression (default: True)
    optimize_data_types (bool): Whether to optimize data types to reduce memory (default: True)
    remove_zero_columns (bool): Whether to remove columns that are all zeros (default: False)
    downsample_factor (int): Factor to downsample data (1=no downsampling, 2=every 2nd sample, etc.)
    auto_detect_format (bool): Whether to auto-detect new FDR format vs old format (default: True)
    """
    
    # Get all CSV files in the folder
    csv_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Check available memory
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)
    print(f"Available memory: {available_gb:.1f} GB")
    
    if len(csv_files) > 500:
        print(f"[WARNING] Processing {len(csv_files)} files - using batch processing")
        batch_size = min(batch_size, 25)  # Smaller batches for large datasets
    
    print(f"Processing in batches of {batch_size} files")
    
    # Determine column names by examining the first file
    first_file = csv_files[0]
    
    # Function to detect header row (skip comment lines starting with #)
    def find_header_row(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if not line.strip().startswith('#') and line.strip():
                        return i
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    for i, line in enumerate(f):
                        if not line.strip().startswith('#') and line.strip():
                            return i
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='cp1252') as f:
                    for i, line in enumerate(f):
                        if not line.strip().startswith('#') and line.strip():
                            return i
        return 0
    
    # Find the header row
    header_row = find_header_row(first_file)
    print(f"Found header row at line {header_row + 1}")
    
    # Read sample with correct header
    try:
        sample_df = pd.read_csv(first_file, nrows=5, encoding='utf-8', skiprows=header_row)
    except UnicodeDecodeError:
        try:
            sample_df = pd.read_csv(first_file, nrows=5, encoding='latin-1', skiprows=header_row)
        except UnicodeDecodeError:
            sample_df = pd.read_csv(first_file, nrows=5, encoding='cp1252', skiprows=header_row)
    
    # Detect format type and set columns accordingly
    if auto_detect_format and 'Time(sec)' in sample_df.columns:
        # New FDR format detected
        print("Detected new FDR format with Time(sec) column")
        timestamp_column = 'Time(sec)'
        
        # For mixed batches, we'll detect per file, so just set a default here
        if 'Value' in sample_df.columns:
            value_column = 'Value'
            data_type = 'numeric'
            print("  - Sample file has NUMERIC data format (using 'Value' column)")
        elif 'TextValue' in sample_df.columns:
            value_column = 'TextValue' 
            data_type = 'discrete'
            print("  - Sample file has DISCRETE data format (using 'TextValue' column)")
        else:
            print("Error: Cannot detect value column in new FDR format")
            return
        
        print("  - Will auto-detect data type for each file during processing")
    else:
        # Old format - use first column as timestamp, second as value
        print("Detected old format - using column positions")
        if len(sample_df.columns) < 2:
            print(f"Error: Files must have at least 2 columns. Found only {len(sample_df.columns)} columns.")
            return
        
        timestamp_column = sample_df.columns[0]
        value_column = sample_df.columns[1]
        data_type = 'legacy'
    
    print(f"\nAuto-detected columns:")
    print(f"  - Timestamp column: '{timestamp_column}'")
    print(f"  - Value column: '{value_column}'")
    print(f"  - Data type: {data_type}")
    
    # Show a preview of the first file structure
    print(f"\nSample data from {os.path.basename(first_file)}:")
    print(sample_df.head(3).to_string())
    
    print(f"\nPhase 2: Processing {len(csv_files)} files...")
    print(f"Using '{value_column}' as the value column")
    print(f"Using '{timestamp_column}' as the timestamp column")
    
    # Initialize dictionary to store all data - much faster than repeated merges
    all_data = {}
    all_timestamps = set()
    
    # Process files in batches to manage memory
    for batch_start in range(0, len(csv_files), batch_size):
        batch_end = min(batch_start + batch_size, len(csv_files))
        batch_files = csv_files[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(csv_files)-1)//batch_size + 1} "
              f"({len(batch_files)} files)")
        
        # Process each file in the batch
        for csv_file in tqdm(batch_files, desc="Processing files"):
            try:
                # Find header row for this file (skip comment lines)
                def find_header_row_for_file(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for i, line in enumerate(f):
                                if not line.strip().startswith('#') and line.strip():
                                    return i
                    except UnicodeDecodeError:
                        try:
                            with open(file_path, 'r', encoding='latin-1') as f:
                                for i, line in enumerate(f):
                                    if not line.strip().startswith('#') and line.strip():
                                        return i
                        except UnicodeDecodeError:
                            with open(file_path, 'r', encoding='cp1252') as f:
                                for i, line in enumerate(f):
                                    if not line.strip().startswith('#') and line.strip():
                                        return i
                    return 0
                
                file_header_row = find_header_row_for_file(csv_file)
                
                # Read the CSV file with UTF-8 encoding and skip comment rows
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8', skiprows=file_header_row)
                except UnicodeDecodeError:
                    # Try alternative encodings if UTF-8 fails
                    try:
                        df = pd.read_csv(csv_file, encoding='latin-1', skiprows=file_header_row)
                    except UnicodeDecodeError:
                        df = pd.read_csv(csv_file, encoding='cp1252', skiprows=file_header_row)
                
                # Get the parameter name based on format
                if timestamp_column == 'Time(sec)':  # New FDR format
                    # Extract parameter name after "_9_"
                    base_filename = Path(csv_file).stem
                    if "_9_" in base_filename:
                        param_name = base_filename.split("_9_")[-1]
                    else:
                        param_name = base_filename
                    
                    # Detect data type for this specific file
                    if 'Value' in df.columns:
                        current_value_column = 'Value'
                        current_data_type = 'numeric'
                    elif 'TextValue' in df.columns:
                        current_value_column = 'TextValue'
                        current_data_type = 'discrete'
                    else:
                        print(f"    [WARNING] No suitable value column found in {param_name}, skipping...")
                        continue
                else:
                    # Legacy format: use full filename
                    param_name = Path(csv_file).stem
                    current_value_column = value_column
                    current_data_type = data_type
                
                # Check if required columns exist
                if timestamp_column not in df.columns:
                    print(f"    [WARNING] '{timestamp_column}' column not found in {param_name}, skipping...")
                    continue
                
                if current_value_column not in df.columns:
                    print(f"    [WARNING] '{current_value_column}' column not found in {param_name}, skipping...")
                    continue
                
                # Extract only the required columns
                file_data = df[[timestamp_column, current_value_column]].copy()
                
                # Convert timestamp to numeric for proper sorting/merging
                timestamps = pd.to_numeric(file_data[timestamp_column], errors='coerce')
                values = file_data[current_value_column]
                
                # Remove rows with invalid timestamps
                valid_mask = ~timestamps.isna()
                timestamps = timestamps[valid_mask]
                values = values[valid_mask]
                
                if len(timestamps) == 0:
                    print(f"    [WARNING] No valid timestamps in {param_name}, skipping...")
                    continue
                
                # Handle value column optimization based on data type
                original_values = values.copy()
                
                if current_data_type == 'discrete':
                    # For discrete data (TextValue), keep as-is (no numeric conversion)
                    processed_values = values
                else:
                    # For numeric data or legacy, try to convert to numeric and optimize
                    try:
                        numeric_values = pd.to_numeric(values, errors='coerce')
                        if not numeric_values.isna().all():
                            # Round only the numeric values
                            mask = ~numeric_values.isna()
                            values = values.copy()
                            values.loc[mask] = numeric_values[mask].round(precision)
                            
                            # Optimize data types if requested
                            if optimize_data_types and not numeric_values.isna().all():
                                min_val = numeric_values.min()
                                max_val = numeric_values.max()
                                
                                if not (pd.isna(min_val) or pd.isna(max_val)):
                                    if min_val >= 0 and max_val <= 255:
                                        values = pd.to_numeric(values, errors='coerce').astype('uint8')
                                    elif min_val >= -128 and max_val <= 127:
                                        values = pd.to_numeric(values, errors='coerce').astype('int8')
                                    elif min_val >= 0 and max_val <= 65535:
                                        values = pd.to_numeric(values, errors='coerce').astype('uint16')
                                    elif min_val >= -32768 and max_val <= 32767:
                                        values = pd.to_numeric(values, errors='coerce').astype('int16')
                                    elif min_val >= -2147483648 and max_val <= 2147483647:
                                        values = pd.to_numeric(values, errors='coerce').astype('int32')
                                    else:
                                        values = pd.to_numeric(values, errors='coerce').astype('float32')
                        processed_values = values
                    except Exception:
                        processed_values = original_values
                
                # Store data in dictionary - much faster than repeated DataFrame merges
                all_data[param_name] = dict(zip(timestamps.astype(int), processed_values))
                all_timestamps.update(timestamps.astype(int))
                
                # Debug: Show sample of data being processed
                non_zero_count = 0
                zero_count = 0
                try:
                    if current_data_type == 'discrete':
                        # For discrete data, count unique values
                        unique_values = processed_values.nunique()
                        print(f"    [OK] {param_name} ({current_data_type}): {len(timestamps)} rows, {unique_values} unique values")
                    else:
                        # For numeric data, count zeros and non-zeros
                        numeric_check = pd.to_numeric(processed_values, errors='coerce')
                        zero_count = (numeric_check == 0).sum()
                        non_zero_count = (~numeric_check.isna() & (numeric_check != 0)).sum()
                        print(f"    [OK] {param_name} ({current_data_type}): {len(timestamps)} rows, {zero_count} zeros, {non_zero_count} non-zero values")
                except Exception:
                    print(f"    [OK] {param_name}: {len(timestamps)} rows")
                
                # Clean up memory
                del df, file_data, timestamps, values, processed_values
                gc.collect()
                
            except Exception as e:
                print(f"    [ERROR] Error processing {os.path.basename(csv_file)}: {str(e)}")
                continue
        
        print(f"    [OK] Batch {batch_start//batch_size + 1} completed")
    
    print(f"\nPhase 3: Building final DataFrame efficiently...")
    
    if all_data:
        # Get sorted list of all unique timestamps
        print("  - Sorting timestamps...")
        all_timestamps = sorted(all_timestamps)
        min_timestamp = all_timestamps[0]
        max_timestamp = all_timestamps[-1]
        
        # Detect sampling interval from first few timestamps
        if len(all_timestamps) > 1:
            time_diffs = [all_timestamps[i+1] - all_timestamps[i] for i in range(min(100, len(all_timestamps)-1))]
            time_diffs = [d for d in time_diffs if d > 0]
            sampling_interval = min(time_diffs) if time_diffs else 125
        else:
            sampling_interval = 125
        
        print(f"  - Timestamp range: {min_timestamp} to {max_timestamp}")
        print(f"  - Detected sampling interval: {sampling_interval}")
        
        # Create continuous timestamp range if needed
        if len(all_timestamps) < (max_timestamp - min_timestamp) // sampling_interval:
            print("  - Creating continuous timestamp range...")
            continuous_timestamps = list(range(int(min_timestamp), int(max_timestamp) + int(sampling_interval), int(sampling_interval)))
        else:
            continuous_timestamps = all_timestamps
        
        # Apply downsampling if requested
        if downsample_factor > 1:
            print(f"  - Downsampling by factor of {downsample_factor}...")
            continuous_timestamps = continuous_timestamps[::downsample_factor]
        
        print(f"  - Building DataFrame with {len(continuous_timestamps):,} rows and {len(all_data)} parameters...")
        
        # Pre-allocate arrays for much faster DataFrame construction
        num_rows = len(continuous_timestamps)
        num_cols = len(all_data)
        
        # Create arrays
        data_dict = {'Sample': range(num_rows), 'Timestamp': continuous_timestamps}
        
        # Fill parameter data efficiently
        for i, (param_name, param_data) in enumerate(all_data.items()):
            print(f"    Processing parameter {i+1}/{num_cols}: {param_name}")
            
            # Create array for this parameter
            param_values = []
            last_value = None
            
            for timestamp in continuous_timestamps:
                if timestamp in param_data:
                    last_value = param_data[timestamp]
                    param_values.append(last_value)
                else:
                    # Forward fill or use 0 if no previous value
                    param_values.append(last_value if last_value is not None else 0)
            
            data_dict[param_name] = param_values
        
        # Create DataFrame from dictionary - much faster than merging
        print("  - Creating final DataFrame...")
        final_df = pd.DataFrame(data_dict)
        
        # Define parameter columns (all columns except Sample and Timestamp)
        parameter_columns = [col for col in final_df.columns if col not in ['Sample', 'Timestamp']]
        
        # Optimize data types for final dataframe if requested
        if optimize_data_types:
            print("  - Optimizing data types to reduce memory usage...")
            for col in parameter_columns:
                if final_df[col].dtype == np.float64:
                    final_df[col] = final_df[col].astype(np.float32)
                elif final_df[col].dtype == np.int64:
                    final_df[col] = final_df[col].astype(np.int32)
        
        # Save the combined CSV (with optional compression)
        output_path = os.path.join(folder_path, output_filename)
        
        # Adjust output filename for compression
        if use_compression and not output_filename.endswith('.gz'):
            if output_filename.endswith('.csv'):
                output_path = output_path[:-4] + '.csv.gz'
            else:
                output_path = output_path + '.gz'
        
        print(f"  - Saving {'with gzip compression' if use_compression else 'without compression'}...")
        
        if use_compression:
            final_df.to_csv(output_path, index=False, compression='gzip', encoding='utf-8')
        else:
            final_df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Calculate file size and show size comparison
        file_size_mb = os.path.getsize(output_path) / (1024**2)
        
        # Estimate uncompressed size if compressed
        if use_compression:
            # Rough estimate: gzip compression typically achieves 3-10x compression
            estimated_uncompressed_mb = file_size_mb * 5  # Conservative estimate
            print(f"\n[OK] Combined CSV saved as: {os.path.basename(output_path)}")
            print(f"[OK] Compressed file size: {file_size_mb:.1f} MB")
            print(f"[OK] Estimated uncompressed size: ~{estimated_uncompressed_mb:.1f} MB")
            print(f"[OK] Compression ratio: ~{estimated_uncompressed_mb/file_size_mb:.1f}x smaller")
        else:
            print(f"\n[OK] Combined CSV saved as: {os.path.basename(output_path)}")
            print(f"[OK] File size: {file_size_mb:.1f} MB")
            
        print(f"[OK] Total rows: {len(final_df):,}")
        print(f"[OK] Total columns: {len(final_df.columns)}")
        print(f"[OK] Sample range: 0 to {len(final_df)-1:,}")
        
        if downsample_factor > 1:
            original_rows = len(final_df) * downsample_factor
            print(f"[OK] Downsampled from {original_rows:,} to {len(final_df):,} rows ({100/downsample_factor:.1f}% of original)")
        
        if remove_zero_columns and 'columns_to_remove' in locals():
            print(f"[OK] Removed {len(columns_to_remove)} zero-only columns")
        
        # Show a preview of the combined data
        print(f"\nPreview of combined data:")
        print(final_df.head(10).to_string())
        
        # Show statistics about data completeness
        print(f"\nData completeness:")
        for col in parameter_columns:
            if final_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                missing_count = final_df[col].isna().sum()
                zero_count = (final_df[col] == 0).sum()
                non_zero_count = ((final_df[col] != 0) & (~final_df[col].isna())).sum()
                print(f"  - {col}: {non_zero_count} non-zero, {zero_count} zeros, {missing_count} missing")
            else:
                empty_count = (final_df[col] == '').sum()
                non_empty_count = (final_df[col] != '').sum()
                print(f"  - {col}: {non_empty_count} non-empty, {empty_count} empty (text column)")
        
        # Remove columns that are all zeros, if requested
        if remove_zero_columns:
            print("  - Removing columns with all zeros...")
            columns_to_remove = []
            
            for col in parameter_columns:
                try:
                    # Check if column is all zeros (much faster on arrays)
                    col_data = final_df[col].values
                    if np.all(col_data == 0):
                        columns_to_remove.append(col)
                except Exception:
                    pass
            
            if columns_to_remove:
                print(f"    Removing {len(columns_to_remove)} zero-only columns")
                final_df = final_df.drop(columns=columns_to_remove)
                # Update parameter_columns after removing columns
                parameter_columns = [col for col in final_df.columns if col not in ['Sample', 'Timestamp']]
        
        # Final data type optimization
        if optimize_data_types:
            print("  - Final data type optimization...")
            for col in parameter_columns:
                if final_df[col].dtype == np.float64:
                    final_df[col] = final_df[col].astype(np.float32)
                elif final_df[col].dtype == np.int64:
                    final_df[col] = final_df[col].astype(np.int32)
        
        return output_path
    else:
        print("No data was successfully processed.")
        return None

def main():
    """
    Main function to run the CSV combiner
    """
    print("=== CSV File Combiner ===")
    print("This program combines multiple CSV files into one.")
    print("\n[FILE SIZE OPTIMIZATION] Features:")
    print("  • Gzip compression: 70-90% size reduction")
    print("  • Data type optimization: 20-50% size reduction") 
    print("  • Remove zero-only columns: Variable reduction")
    print("  • Downsampling: Proportional reduction")
    print("  • All optimizations can be combined for maximum effect")
    print()
    
    # Get the current folder path
    current_folder = os.getcwd()
    
    # Ask user for folder path or use current folder
    folder_input = input(f"Enter folder path (press Enter to use current folder: {current_folder}): ").strip()
    
    if folder_input:
        folder_path = folder_input
    else:
        folder_path = current_folder
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Ask for output filename
    output_name = input("Enter output filename (press Enter for 'combined_data.csv'): ").strip()
    if not output_name:
        output_name = 'combined_data.csv'
    
    # Ensure the output filename has .csv extension
    if not output_name.endswith('.csv'):
        output_name += '.csv'
    
    # Ask for optimization settings
    print("\n=== Processing Settings ===")
    batch_size = input("Enter batch size for processing (default: 100): ").strip()
    batch_size = int(batch_size) if batch_size.isdigit() else 100
    
    precision = input("Enter decimal precision for NUMERIC values only (default: 4): ").strip()
    precision = int(precision) if precision.isdigit() else 4
    
    use_compression = input("Use gzip compression for output file? (y/n, default: y): ").strip().lower()
    use_compression = use_compression == 'y' if use_compression else True
    
    optimize_data_types = input("Optimize data types to reduce memory usage? (y/n, default: y): ").strip().lower()
    optimize_data_types = optimize_data_types == 'y' if optimize_data_types else True
    
    remove_zero_columns = input("Remove columns that are all zeros? (y/n, default: n): ").strip().lower()
    remove_zero_columns = remove_zero_columns == 'y' if remove_zero_columns else False
    
    downsample_factor = input("Downsample data by a factor (default: 1, no downsampling): ").strip()
    downsample_factor = int(downsample_factor) if downsample_factor.isdigit() else 1
    
    print(f"\nSettings:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Precision: {precision} decimal places (numeric values only)")
    print(f"  - Output: {'Gzip compressed CSV' if use_compression else 'Uncompressed CSV'}")
    print(f"  - Optimize data types: {'Yes' if optimize_data_types else 'No'}")
    print(f"  - Remove zero columns: {'Yes' if remove_zero_columns else 'No'}")
    print(f"  - Downsample factor: {downsample_factor}")
    
    # Run the combination process
    try:
        result = combine_csv_files(folder_path, output_filename=output_name, 
                                 batch_size=batch_size, precision=precision,
                                 use_compression=use_compression, 
                                 optimize_data_types=optimize_data_types,
                                 remove_zero_columns=remove_zero_columns,
                                 downsample_factor=downsample_factor,
                                 auto_detect_format=True)
        if result:
            print(f"\n[SUCCESS] Combined file created at: {result}")
        else:
            print("\n[ERROR] Failed to create combined file.")
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
