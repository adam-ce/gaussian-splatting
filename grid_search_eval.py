import pandas as pd

def process_csv(file_path):
    # Read the CSV file without header, assuming no column names
    df = pd.read_csv(file_path, header=None)

    # Get the number of columns
    num_columns = df.shape[1]

    # Group by the first two columns and select the row with the smallest value in the last column
    result = df.loc[df.groupby([0, 1, 2])[num_columns - 1].idxmax()]

    return result

def count_strings_in_columns(result_df):
    # Group by column 1
    grouped = result_df.groupby(1)

    # Count occurrences of each string in columns 6 and 7 within each group
    counts = grouped[6].value_counts().unstack(fill_value=0), grouped[7].value_counts().unstack(fill_value=0)
    
    return counts

# Example usage
file_path = 'grid_search_eval_out_2.csv'
result_df = process_csv(file_path)

# Adjust display settings to see the entire DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(result_df)

counts_col_6, counts_col_7 = count_strings_in_columns(result_df)
print("\nCounts of strings in column 6:")
print(counts_col_6)

print("\nCounts of strings in column 7:")
print(counts_col_7)