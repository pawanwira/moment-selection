#!/usr/bin/env python3
"""
Format transcripts script for moment-selection project.

This script processes CSV transcript files in the unformatted_transcripts/ directory:
1. Renames files to only include the ID (removes _transcript.csv suffix)
2. Creates instance IDs for teaching practice columns
3. [Future] Will format transcripts with additional processing
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from constants import UNFORMATTED_TRANSCRIPTS_DIR, TRANSCRIPTS_DIR, TEACHING_PRACTICE_COL_NAMES


def create_instance_ids(df):
    """
    Create instance IDs for teaching practice columns.
    An instance is consecutive rows with the same teaching practice label.
    Instance IDs start from 1 (1-indexed).
    """
    for col_name in TEACHING_PRACTICE_COL_NAMES:
        if col_name in df.columns:
            instance_col_name = f"{col_name}_instance_id"
            df[instance_col_name] = 0
            
            # Create instance IDs for consecutive rows with the same label
            current_instance = 0
            previous_value = None
            
            for idx, value in enumerate(df[col_name]):
                if pd.notna(value) and value == 1.0:  # Check for 1.0 values (teaching practice present)
                    if previous_value != 1.0:  # Start new instance if previous row wasn't a practice
                        current_instance += 1
                    df.loc[idx, instance_col_name] = current_instance
                previous_value = value  # Track the actual value for next iteration
    
    return df


def main():
    """Main function to process transcript files."""
    # Define directories using constants
    unformatted_dir = Path(UNFORMATTED_TRANSCRIPTS_DIR)
    transcripts_dir = Path(TRANSCRIPTS_DIR)
    
    # Ensure directories exist
    unformatted_dir.mkdir(exist_ok=True)
    transcripts_dir.mkdir(exist_ok=True)
    
    # Process all CSV files in unformatted_transcripts/
    csv_files = list(unformatted_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {UNFORMATTED_TRANSCRIPTS_DIR}/ directory")
        return
    
    print(f"Found {len(csv_files)} CSV files to process:")
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")
        
        # Extract ID from filename (assumes format: ID_transcript.csv)
        if "_transcript.csv" in csv_file.name:
            transcript_id = csv_file.name.replace("_transcript.csv", "")
            new_filename = f"{transcript_id}.csv"
        else:
            # If filename doesn't follow expected pattern, use the base name
            transcript_id = csv_file.stem
            new_filename = csv_file.name
        
        # Read CSV file and create instance IDs
        try:
            df = pd.read_csv(csv_file)
            print(f"  -> Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Create instance IDs for teaching practice columns
            df = create_instance_ids(df)
            
            # Save processed file directly to transcripts directory
            new_file_path = transcripts_dir / new_filename
            df.to_csv(new_file_path, index=False)
            print(f"  -> Saved processed file to: {new_file_path}")
            
        except Exception as e:
            print(f"  -> Error processing {csv_file.name}: {e}")
            # Still copy the original file even if processing fails
            new_file_path = transcripts_dir / new_filename
            shutil.copy2(csv_file, new_file_path)
            print(f"  -> Copied original file to: {new_file_path}")
        
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
