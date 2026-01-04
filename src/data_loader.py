import pandas as pd
import os

def load_and_clean_data(raw_data_path, processed_data_path):
    """
    Loads raw Superstore data, cleans dates, removes returns, and saves to processed folder.
    """
    print("Loading data...")
    
    

    # Load data
    try:
        df_orders = pd.read_excel(raw_data_path, sheet_name= "Orders")
        df_returns = pd.read_excel(raw_data_path, sheet_name= "Returns")
        # People data is often not needed for demand forecasting, but we load it just in case
        df_people = pd.read_excel(raw_data_path, sheet_name= "People")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    # 1. Convert Dates (Excel Serial Format)
    # Excel dates are days since Dec 30, 1899
    # print("Converting dates...")
    # df_orders['Order Date'] = pd.to_datetime(df_orders['Order Date'], unit='D', origin='1899-12-30')
    # df_orders['Ship Date'] = pd.to_datetime(df_orders['Ship Date'], unit='D', origin='1899-12-30')

    # 2. Merge with Returns
    print("Merging returns...")
    df_merged = df_orders.merge(df_returns, on='Order ID', how='left')
    
    # Fill NaN in 'Returned' with 'No'
    df_merged['Returned'] = df_merged['Returned'].fillna('No')

    # 3. Filter out Returns (Keep only Net Sales)
    df_clean = df_merged[df_merged['Returned'] == 'No'].copy()
    
    # Drop the temporary 'Returned' column
    df_clean.drop(columns=['Returned'], inplace=True)
    
    print("Converting dates...")
    df_clean['Order Date'] = pd.to_datetime(df_clean['Order Date'], dayfirst= False)
    df_clean['Ship Date'] = pd.to_datetime(df_clean['Ship Date'], dayfirst= False)

    # basic columns renaming
    df_clean.columns = (df_clean.columns.str.strip().str.lower().str.replace(' ','_').str.replace('-','_'))

    # Sort by Date
    df_clean.sort_values('order_date', inplace=True)

    # 4. Save to Processed
    output_file = os.path.join(processed_data_path, "superstore_clean.csv")
    print(f"Saving processed data to {output_file}...")
    df_clean.to_csv(output_file, index=False)
    
    print(f"Done! Original size: {len(df_orders)}, Clean size: {len(df_clean)}")
    return df_clean

if __name__ == "__main__":
    # Define your paths relative to this script or absolute
    RAW_PATH = "D:/KAL_X/DS PROJECT/Intelligent_Supply_Chain/data/raw/Superstore.xlsx"
    PROCESSED_PATH = "D:/KAL_X/DS PROJECT/Intelligent_Supply_Chain/data/processed"
    
    load_and_clean_data(RAW_PATH, PROCESSED_PATH)