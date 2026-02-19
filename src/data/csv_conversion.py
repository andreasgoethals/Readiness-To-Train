
"""
CSV Conversion Script
Converts Readiness_Data.xlsx to Readiness_Data.csv

Usage:
    python src/data/csv_conversion.py
"""

import pandas as pd
from pathlib import Path


def convert_excel_to_csv():
    """Convert the Excel file to CSV format."""
    
    # Define paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    input_path = project_root / "data" / "raw" / "Readiness_Data.xlsx"
    output_path = project_root / "data" / "raw" / "Readiness_Data.csv"
    
    # Read Excel file
    print(f"Reading: {input_path}")
    df = pd.read_excel(input_path)
    
    # Export to CSV
    df.to_csv(
        output_path,
        index=False,
        sep=',',
        encoding='utf-8',
        date_format='%Y-%m-%d',
        quoting=1  # Quote non-numeric fields (handles comma in 'Max, Velocity%')
    )
    
    print(f"Saved:   {output_path}")
    print(f"Shape:   {df.shape[0]} rows × {df.shape[1]} columns")


if __name__ == "__main__":
    convert_excel_to_csv()