import pandas as pd
import numpy as np

def analyze_column_distribution(df, column_name, weight_column='hv005', is_time_column=False, time_bins=None):
    """
    Analyzes the distribution of values in a specified column, providing
    unweighted and weighted counts and percentages.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to analyze.
        weight_column (str): The name of the weight column (default: 'hv005').
        is_time_column (bool): If True, treats the column as time values and applies binning.
        time_bins (list): A list of bin edges for time columns (e.g., [0, 10, 20, 30, 60, np.inf]).
                          If None, default bins will be used.
    """
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in the DataFrame.")
        return
    if weight_column not in df.columns:
        print(f"Error: Weight column '{weight_column}' not found in the DataFrame. Cannot compute weighted stats.")
        return

    print(f"\n--- Analysis for Column: '{column_name}' ---")

    # --- Prepare the Series ---
    series = df[column_name].copy()
    weights = df[weight_column].copy()

    # Handle missing values by converting to object (allowing mixed types) then filling with 'Missing' string
    # This ensures 'Missing' is treated as a distinct category.
    series_for_analysis = series.astype(object).fillna('Missing')
    
    # Handle missing values in weights (though usually weights don't have NaNs)
    valid_indices = weights.notna()
    series_for_analysis = series_for_analysis[valid_indices]
    weights = weights[valid_indices]

    # --- Binning for Time Columns ---
    if is_time_column:
        if time_bins is None:
            time_bins = [0, 10, 20, 30, 60, 120, 240, np.inf] # Default bins for time (in minutes)
        
        # Isolate non-missing values for numeric conversion and binning
        non_missing_series_idx = series_for_analysis != 'Missing'
        temp_series_no_missing = series_for_analysis[non_missing_series_idx].copy()
        
        # Convert to numeric, coercing errors to NaN. This will turn non-numeric strings into NaN.
        numeric_series = pd.to_numeric(temp_series_no_missing, errors='coerce')
        
        # Create bins for the numeric values.
        # If all values became NaN after coercion, pd.cut will raise ValueError.
        # Handle this by creating a series of 'Uncategorized Numeric' for those.
        try:
            binned_series = pd.cut(numeric_series, bins=time_bins, right=False, include_lowest=True).astype(object)
            # Mark values that were numeric but didn't fit any bin (e.g., NaN from coercion)
            binned_series[numeric_series.isna()] = 'Uncategorized Numeric'
        except ValueError: # This happens if numeric_series is empty or all NaN for pd.cut
            binned_series = pd.Series('Uncategorized Numeric', index=numeric_series.index, dtype=object)

        # Reconstruct the full series, combining binned numeric values with 'Missing' values
        final_series_processed = pd.Series(index=series_for_analysis.index, dtype=object)
        final_series_processed[non_missing_series_idx] = binned_series.reindex(series_for_analysis[non_missing_series_idx].index)
        final_series_processed[series_for_analysis == 'Missing'] = 'Missing'
        
        series_to_process = final_series_processed
    else:
        # For non-time columns, ensure consistent type for grouping by converting to string
        # This handles cases where original data might be mixed (e.g., int codes and string labels)
        series_to_process = series_for_analysis.astype(str) # Convert everything to string for safe grouping/sorting

    # --- Unweighted Counts ---
    unweighted_counts = series_to_process.value_counts(dropna=False).sort_index()
    unweighted_percentages = (unweighted_counts / unweighted_counts.sum() * 100).round(2)

    print("\n--- Unweighted Distribution ---")
    unweighted_df = pd.DataFrame({
        'Count': unweighted_counts,
        'Percentage (%)': unweighted_percentages
    })
    print(unweighted_df.to_string())

    # --- Weighted Counts ---
    # Group by the series values and sum the weights
    # series_to_process is now guaranteed to be 'str' type, which handles sorting consistently
    weighted_counts = series_to_process.groupby(series_to_process).apply(lambda x: weights[x.index].sum())
    # Sort by index (value) for consistency - this will now sort strings lexicographically
    weighted_counts = weighted_counts.sort_index()
    
    # Calculate weighted percentages
    total_weighted_sum = weights.sum()
    weighted_percentages = (weighted_counts / total_weighted_sum * 100).round(2)

    print("\n--- Weighted Distribution ---")
    weighted_df = pd.DataFrame({
        'Weighted Count': weighted_counts,
        'Weighted Percentage (%)': weighted_percentages
    })
    print(weighted_df.to_string())
    print(f"\nTotal Unweighted Records: {len(series)}")
    print(f"Total Weighted Sum: {total_weighted_sum:,.2f}")
    print("-" * 50)


# --- Main Program ---
if __name__ == "__main__":
    # --- Configuration ---
    DTA_FILE_PATH = '/Users/kumar.aniket/Downloads/surveydata/IAHR7EFL.DTA'  # Replace with your actual path
    WEIGHT_COLUMN = 'hv005'       # The column containing the household weight

    # Define columns to analyze and their types
    columns_to_analyze_config = [
        {'name': 'hv025', 'type': 'categorical'},  # Residence (Urban/Rural)
        {'name': 'hv270', 'type': 'categorical'},  # Wealth index
        {'name': 'sh37b', 'type': 'categorical'},  # Water disruption (Yes/No)
        {'name': 'hv204', 'type': 'time', 'bins': [0, 15, 30, 60, 120, np.inf]}, # Time to water source
        {'name': 'hv201', 'type': 'categorical'},  # Source of drinking water
        {'name': 'hv009', 'type': 'numeric'},      # Household size
        {'name': 'hv014', 'type': 'numeric'},      # Children under 5
        {'name': 'hv236', 'type': 'categorical'},  # Person who fetches water
        {'name': 'hv206', 'type': 'categorical'},  # Has electricity
        {'name': 'hv205', 'type': 'categorical'},  # Type of toilet facility
        {'name': 'hv006', 'type': 'categorical'}   # Month of interview
    ]

    try:
        # Collect all unique column names needed
        required_columns = [col['name'] for col in columns_to_analyze_config]
        if WEIGHT_COLUMN not in required_columns:
            required_columns.append(WEIGHT_COLUMN) # Ensure weight column is always loaded

        # Load only the necessary columns
        print(f"Loading data from '{DTA_FILE_PATH}' with selected columns: {required_columns}")
        # Use low_memory=False for potentially mixed types in large files, though 'columns' param helps
        df = pd.read_stata(DTA_FILE_PATH, columns=required_columns, convert_categoricals=False)
        print("Data loaded successfully.")
        print(f"DataFrame shape (loaded columns only): {df.shape}")
        
        # Ensure the weight column is numeric and handle potential NaNs
        df[WEIGHT_COLUMN] = pd.to_numeric(df[WEIGHT_COLUMN], errors='coerce').fillna(0)
        
        # Filter out rows where weight is zero or NaN, as they won't contribute to weighted stats
        initial_rows = df.shape[0]
        df_filtered = df[df[WEIGHT_COLUMN] > 0].copy()
        print(f"Filtered out {initial_rows - df_filtered.shape[0]} rows with zero or NaN weights.")
        print(f"DataFrame shape after weight filtering: {df_filtered.shape}")

        # Analyze each specified column
        for col_info in columns_to_analyze_config:
            col_name = col_info['name']
            col_type = col_info['type']
            custom_bins = col_info.get('bins', None)

            if col_type == 'time':
                analyze_column_distribution(df_filtered, col_name, weight_column=WEIGHT_COLUMN, 
                                            is_time_column=True, time_bins=custom_bins)
            else:
                analyze_column_distribution(df_filtered, col_name, weight_column=WEIGHT_COLUMN)

    except FileNotFoundError:
        print(f"Error: The file '{DTA_FILE_PATH}' was not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")

