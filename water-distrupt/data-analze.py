import pandas as pd
import numpy as np

def analyze_column(file_path, target_column, weight_column='hv005'):
    """
    Load NHFS .dta file and display unweighted and weighted value counts for a specific column
    
    Parameters:
    file_path (str): Path to the .dta file
    target_column (str): Column name to analyze
    weight_column (str): Column name for weights (default: 'hv005')
    """
    try:
        # Load the .dta file with only required columns
        df = pd.read_stata(file_path, columns=[target_column, weight_column], convert_categoricals=False)
        
        # Try to load value labels
        try:
            with pd.io.stata.StataReader(file_path) as reader:
                value_labels = reader.value_labels()
        except:
            value_labels = None
        
        # Check if target column exists (try different case variations)
        possible_columns = [target_column, target_column.lower(), target_column.upper(),
                          f'H{target_column[1:]}', f'h{target_column[1:]}']
        
        found_column = None
        for col in possible_columns:
            if col in df.columns:
                found_column = col
                break
        
        if found_column is None:
            print(f"❌ Column '{target_column}' not found!")
            return None
        
        # Verify weight column
        if weight_column not in df.columns:
            print(f"❌ Weight column '{weight_column}' not found!")
            return None
        
        # Get unweighted value counts and percentages
        column_data = df[found_column]
        value_counts = column_data.value_counts(dropna=False)
        total = len(column_data)
        value_percentages = (value_counts / total * 100).round(1)
        
        # Get weighted value counts and percentages
        weighted_counts = df.groupby(found_column)[weight_column].sum()
        total_weighted = weighted_counts.sum()
        weighted_percentages = (weighted_counts / total_weighted * 100).round(1)
        
        # Get labels if available
        labels = value_labels.get(found_column, {}) if value_labels else {}
        
        # Print unweighted results
        print(f"Variable {found_column} found (unweighted):")
        for value, count in value_counts.items():
            percentage = value_percentages[value]
            if pd.isna(value):
                label = "Code nan"
            else:
                label = labels.get(value, str(value)).lower()
            print(f"  {label}: {count:,} ({percentage}%)")
        print(f"Total: {total:,}")
        
        # Print weighted results
        print(f"\nVariable {found_column} found (weighted):")
        for value, count in weighted_counts.items():
            percentage = weighted_percentages[value]
            if pd.isna(value):
                label = "Code nan"
            else:
                label = labels.get(value, str(value)).lower()
            print(f"  {label}: {int(count):,} ({percentage}%)")
        print(f"Total (weighted): {int(total_weighted):,}")
        
        return df, found_column, value_counts, weighted_counts
    
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

if __name__ == "__main__":
    file_path = "/Users/kumar.aniket/Downloads/surveydata/IAHR7EFL.DTA"
    target_column = 'sh37b'
    weight_column = 'hv005'
    result = analyze_column(file_path, target_column, weight_column)
    
    if result is None:
        print("\n❌ Analysis could not be completed. Please check the file path, column name, or weight column.")
    else:
        print(f"\n✅ Successfully analyzed column '{target_column}' with weights from '{weight_column}'")