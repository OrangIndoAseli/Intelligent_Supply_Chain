import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import os
import sys
import itertools
import logging

# Suppress Prophet logs to keep terminal clean
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# Ensure we can import from src
sys.path.append(os.getcwd())

# --- CONFIG ---
DATA_PATH = "data/processed/superstore_clean.csv"
OUTPUT_REPORT = "reports/category_restocking_plan.csv"
FIGURES_DIR = "reports/figures/"

# Create figures dir if not exists
os.makedirs(FIGURES_DIR, exist_ok=True)

# PARAMETER GRID (The "Search Space")
# We test these combinations for EACH category
PARAM_GRID = {
    'changepoint_prior_scale': [0.01, 0.1, 0.5],
    'seasonality_prior_scale': [1.0, 10.0],
}

SERVICE_LEVEL_Z = 1.65  # 95% Confidence

def find_best_params(df_category):
    """
    Runs a Grid Search to find the best Prophet parameters for a specific dataframe.
    Returns: (best_params, best_rmse)
    """
    all_params = [dict(zip(PARAM_GRID.keys(), v)) for v in itertools.product(*PARAM_GRID.values())]
    best_params = None
    best_rmse = float('inf')
    
    # We need enough data for CV. If dataset is tiny, skip grid search and use defaults.
    if len(df_category) < 24: # Less than 2 years of data
        return {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0}, 0.0

    print(f"   > Tuning... testing {len(all_params)} combinations...")
    
    for params in all_params:
        m = Prophet(**params, seasonality_mode='multiplicative')
        m.add_country_holidays(country_name='US')
        m.fit(df_category)
        
        try:
            # Quick CV: 2 years initial, forecast 1 month ahead
            df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='30 days', disable_tqdm=True)
            df_p = performance_metrics(df_cv)
            rmse = df_p['rmse'].mean()
        except:
            rmse = float('inf')
            
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            
    return best_params, best_rmse

def run_category_forecast():
    print("===================================================")
    print("   INTELLIGENT SUPPLY CHAIN: OPTIMIZED ENGINE      ")
    print("===================================================")

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    df['order_date'] = pd.to_datetime(df['order_date'])

    categories = df['category'].unique()
    all_recommendations = []

    for category in categories:
        print(f"\n[PROCESSING] Category: {category}...")
        
        # 2. Filter & Resample
        cat_df = df[df['category'] == category].copy()
        monthly_sales = cat_df.set_index('order_date').resample('MS')['sales'].sum().reset_index()
        monthly_sales.columns = ['ds', 'y']
        
        # 3. AUTO-TUNE: Find Best Parameters
        best_params, best_rmse = find_best_params(monthly_sales)
        
        print(f"   > Winner: {best_params}")
        print(f"   > RMSE: ${best_rmse:,.2f}")

        # 4. Train Final Model with Winner Params
        model = Prophet(**best_params, seasonality_mode='multiplicative')
        model.add_country_holidays(country_name='US')
        model.fit(monthly_sales)

        # 5. Forecast Future
        future = model.make_future_dataframe(periods=12, freq='MS')
        forecast = model.predict(future)
        
        # 6. VISUALIZATION
        plt.figure(figsize=(12, 6))
        
        # Actuals
        plt.plot(monthly_sales['ds'], monthly_sales['y'], 'ko-', label='Actual History')
        
        # Forecast
        future_only = forecast[forecast['ds'] > monthly_sales['ds'].max()]
        plt.plot(forecast['ds'], forecast['yhat'], 'b-', label='Model Fit')
        plt.plot(future_only['ds'], future_only['yhat'], 'g--', linewidth=2, label='Future Forecast')
        
        # Uncertainty Interval
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.1)
        
        plt.title(f"Optimized Forecast: {category}\n(RMSE: ${best_rmse:.0f})", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Sales ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save Plot
        plot_filename = os.path.join(FIGURES_DIR, f"forecast_{category.replace(' ', '_')}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"   > Plot saved to: {plot_filename}")

        # 7. Calculate Advice
        next_month_pred = future_only.iloc[0]['yhat']
        safety_stock = best_rmse * SERVICE_LEVEL_Z
        order_amount = next_month_pred + safety_stock
        
        all_recommendations.append({
            'Category': category,
            'Forecast_Month': future_only.iloc[0]['ds'].strftime('%Y-%m'),
            'Predicted_Demand': round(next_month_pred, 2),
            'Model_RMSE': round(best_rmse, 2),
            'Safety_Stock_Buffer': round(safety_stock, 2),
            'Recommended_Order': round(order_amount, 2)
        })

    # 8. Save Final Report
    results_df = pd.DataFrame(all_recommendations)
    results_df.to_csv(OUTPUT_REPORT, index=False)
    
    print("\n===================================================")
    print("           FINAL RESTOCKING ADVICE                 ")
    print("===================================================")
    print(results_df[['Category', 'Predicted_Demand', 'Recommended_Order']].to_string(index=False))

if __name__ == "__main__":
    run_category_forecast()