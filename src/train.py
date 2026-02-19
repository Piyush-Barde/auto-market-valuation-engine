import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import sys
import os
import warnings

# --- CONFIGURATION ---
CSV_FILE_PATH = r'C:\Users\vishwa\Downloads\cardekho.csv'
CURRENT_YEAR = 2026
RANDOM_STATE = 42
warnings.filterwarnings('ignore')

def load_and_clean_data(filepath):
    if not os.path.exists(filepath):
        sys.exit(f"[CRITICAL ERROR] File not found at: {filepath}")

    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower()
        
        # 1. robust column renaming
        if 'name' not in df.columns:
            for p in ['car_name', 'model', 'title']:
                if p in df.columns: 
                    df.rename(columns={p: 'name'}, inplace=True)
                    break
            else:
                sys.exit("CRITICAL ERROR: 'name' column not found.")

        rename_map = {
            'year': 'Year',
            'selling_price': 'Seller_Price', 
            'km_driven': 'Kilometers',
            'fuel': 'Fuel',
            'transmission': 'Transmission',
            'owner': 'Owner',
            'max_power': 'Max_Power',
        }
        df = df.rename(columns=rename_map)
        df = df.dropna()
        
        # 2. Extract Brand and Model
        df['Brand'] = df['name'].apply(lambda x: str(x).split()[0])
        df['Model'] = df['name'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else x.split()[0])
        df['Car_Age'] = CURRENT_YEAR - df['Year']
        
        # 3. Clean Power Column
        def clean_power(x):
            try:
                return float(str(x).lower().replace('bhp', '').replace('ps', '').strip())
            except:
                return np.nan
        
        if 'Max_Power' in df.columns:
            df['Max_Power'] = df['Max_Power'].apply(clean_power)
            df = df.dropna(subset=['Max_Power'])

        return df
    except Exception as e:
        sys.exit(f"Error reading file: {e}")

def add_original_price_column(df):
    """
    Adds a new column 'Original_Purchase_Price'.
    Reverse-engineers using 11% annual depreciation.
    """
    DEPRECIATION_RATE = 0.11
    
    def calculate_new_price(row):
        age = row['Car_Age']
        current_price = row['Seller_Price']
        if age <= 0: return current_price
        # Reverse Compound Interest Formula
        return current_price / ((1 - DEPRECIATION_RATE) ** age)

    df['Original_Purchase_Price'] = df.apply(calculate_new_price, axis=1)
    return df

def train_valuation_model(df):
    print(f">> Training Market Model on {len(df)} cars... (Please wait)")
    feature_cols = ['Brand', 'Year', 'Car_Age', 'Fuel', 'Transmission', 'Max_Power', 'Kilometers', 'Owner']
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].copy()
    y = df['Seller_Price']
    
    encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X, y)
    
    return model, encoders, feature_cols

def get_user_input():
    print("\n" + "="*50)
    print("      USED CAR MARKET ANALYZER")
    print("="*50)
    try:
        p_min = float(input("Min Budget (₹) [0]: ") or 0)
        p_max = float(input("Max Budget (₹) [No Limit]: ") or float('inf'))
        y_min = int(input("Min Year [2014]: ") or 2014)
        brand = input("Brand (e.g. Honda) [Enter for All]: ").strip()
        return p_min, p_max, y_min, brand
    except Exception as e:
        print(f"Invalid Input: {e}")
        return 0, float('inf'), 2014, ""

def analyze_deals(df, model, encoders, feats, min_p, max_p, min_y, brand):
    # 1. Apply Filters
    mask = (df['Seller_Price'] >= min_p) & \
            (df['Seller_Price'] <= max_p) & \
            (df['Year'] >= min_y)
    
    if brand: 
        mask = mask & df['Brand'].str.contains(brand, case=False)
    
    res = df[mask].copy()
    if res.empty: return None
    
    # 2. Predict Fair Market Value
    X_pred = res[feats].copy()
    for col, le in encoders.items():
        X_pred[col] = X_pred[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
        X_pred[col] = le.transform(X_pred[col])
        
    res['Fair_Market_Value'] = model.predict(X_pred)
    res['Savings'] = res['Fair_Market_Value'] - res['Seller_Price']
    
    # 3. Verdict Logic
    # 5% tolerance margin
    margin = res['Fair_Market_Value'] * 0.05
    
    conditions = [
        (res['Savings'] > margin),        # Saving Money
        (res['Savings'] < -margin),       # Losing Money
    ]
    choices = ['GOOD DEAL', 'BAD DEAL']
    
    # Default is now FAIR DEAL
    res['Verdict'] = np.select(conditions, choices, default='FAIR DEAL')
    
    return res

def main():
    # 1. Load Data
    df = load_and_clean_data(CSV_FILE_PATH)
    
    # 2. Add Original Price
    df = add_original_price_column(df)
    
    # 3. Train AI Model
    model, encoders, feats = train_valuation_model(df)
    
    # 4. Get Input
    min_p, max_p, min_y, brand = get_user_input()
    
    # 5. Analyze
    print(f"\n--- Analyzing Market for {brand if brand else 'All Brands'} ---")
    results = analyze_deals(df, model, encoders, feats, min_p, max_p, min_y, brand)
    
    if results is None:
        print("\n[!] No cars found. Try widening your search.")
        return

    # 6. SHOW DATA TABLE
    pd.options.display.float_format = '{:,.0f}'.format
    
    # ADDED 'Fair_Market_Value' to display columns
    display_cols = ['name', 'Year', 'Original_Purchase_Price', 'Fair_Market_Value', 'Seller_Price', 'Verdict']
    
    print("\n" + "="*100)
    print(f"REPORT: {len(results)} cars found.")
    print("Fair_Market_Value = AI Estimated Value | Verdict = Is the Seller asking too much?")
    print("="*100)
    
    sorted_res = results.sort_values('Savings', ascending=False)
    
    print("\n--- TOP 5 GOOD DEALS (Undervalued) ---")
    print(sorted_res[sorted_res['Verdict'] == 'GOOD DEAL'].head(5)[display_cols].to_string(index=False))

    # ADDED FAIR DEALS SECTION
    print("\n--- TOP 5 FAIR DEALS (Priced Correctly) ---")
    fair_deals = sorted_res[sorted_res['Verdict'] == 'FAIR DEAL']
    if not fair_deals.empty:
        print(fair_deals.head(5)[display_cols].to_string(index=False))
    else:
        print("No 'Fair Deal' listings found.")

    print("\n--- TOP 5 BAD DEALS (Overpriced) ---")
    print(sorted_res[sorted_res['Verdict'] == 'BAD DEAL'].tail(5)[display_cols].to_string(index=False))

    # 7. GENERATE 3-BAR GRAPH
    print("\nGenerating Comparison Graph...")
    
    if brand:
        group_col = 'Model'
        title = f"Model Valuation ({brand})"
    else:
        group_col = 'Brand'
        title = "Brand Valuation (Top 10)"

    # Aggregate Data
    top_cats = results[group_col].value_counts().head(10).index
    plot_data = results[results[group_col].isin(top_cats)]
    
    # Group and Calculate Means for ALL THREE METRICS
    grouped = plot_data.groupby(group_col)[['Original_Purchase_Price', 'Fair_Market_Value', 'Seller_Price']].mean().reset_index()
    
    # Melt
    df_melted = grouped.melt(id_vars=group_col, 
                                value_vars=['Original_Purchase_Price', 'Fair_Market_Value', 'Seller_Price'], 
                                var_name='Price_Type', value_name='Price')

    # Plot
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    
    bar_plot = sns.barplot(
        data=df_melted, 
        x=group_col, 
        y='Price', 
        hue='Price_Type',
        palette={
            'Original_Purchase_Price': 'gray',       # New
            'Fair_Market_Value': '#2ca02c',          # Fair (Green)
            'Seller_Price': '#d62728'                # Seller (Red)
        }
    )
    
    plt.title(f'Valuation Reality: {title}', fontsize=16, fontweight='bold')
    plt.ylabel('Average Price (₹)', fontsize=12)
    plt.xlabel(group_col, fontsize=12)
    
    # Format Y-Axis
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f'{x/100000:.1f}L' for x in current_values])
    
    # Legend
    handles, labels = bar_plot.get_legend_handles_labels()
    plt.legend(handles, ['Original Price (New)', 'Fair Market Value', 'Seller Asking Price'])
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()