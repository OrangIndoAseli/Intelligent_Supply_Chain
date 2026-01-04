import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Supply Chain Command Center",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIGURATION ---
DATA_PATH = "data/processed/superstore_clean.csv"

# Robust Parameters found during our Optimization Step
# We use these defaults for the 'Live' app to ensure speed
MODEL_PARAMS = {
    'changepoint_prior_scale': 0.01, 
    'seasonality_prior_scale': 10.0,
    'seasonality_mode': 'multiplicative'
}
SERVICE_LEVEL_Z = 1.65  # 95% Confidence

# --- DATA LOADING (Cached) ---
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at {DATA_PATH}. Please run data_loader.py first.")
        return None
    
    df = pd.read_csv(DATA_PATH)
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['YearMonth'] = df['order_date'].dt.to_period('M').astype(str)
    return df

df = load_data()

# --- MODEL ENGINE (Cached) ---
@st.cache_data(show_spinner="Training AI Model & Generating Forecast...")
def run_forecast(category, _df_full):
    """
    Trains a Prophet model on the fly for the selected Category.
    """
    # 1. Filter Data
    cat_df = _df_full[_df_full['category'] == category].copy()
    monthly_sales = cat_df.set_index('order_date').resample('MS')['sales'].sum().reset_index()
    monthly_sales.columns = ['ds', 'y']
    
    # 2. Train Model
    m = Prophet(**MODEL_PARAMS)
    m.add_country_holidays(country_name='US')
    m.fit(monthly_sales)
    
    # 3. Forecast
    future = m.make_future_dataframe(periods=12, freq='MS')
    forecast = m.predict(future)
    
    # 4. Calculate Error (Quick approximation for speed)
    # In production, we might load pre-calculated RMSEs to save time
    # Here we use the standard deviation of residuals for a fast "live" estimate
    residuals = monthly_sales['y'] - forecast.iloc[:len(monthly_sales)]['yhat']
    rmse = (residuals ** 2).mean() ** 0.5
    
    return monthly_sales, forecast, rmse

# --- SIDEBAR ---
st.sidebar.title("Command Center")
page = st.sidebar.radio("Navigate", ["📊 Business Dashboard", "🔮 AI Forecaster"])
st.sidebar.markdown("---")
st.sidebar.info("Use the 'AI Forecaster' to generate restocking plans for specific categories.")

# ==========================================
# PAGE 1: BUSINESS DASHBOARD
# ==========================================
if page == "📊 Business Dashboard":
    st.title("📊 Supply Chain Performance Dashboard")
    st.markdown("### Historical Performance Overview")

    if df is not None:
        # --- KPI ROW ---
        col1, col2, col3, col4 = st.columns(4)
        
        total_sales = df['sales'].sum()
        total_profit = df['profit'].sum()
        total_orders = df['order_id'].nunique()
        profit_margin = (total_profit / total_sales) * 100

        col1.metric("Total Sales", f"${total_sales:,.0f}", delta="vs 4 yr avg")
        col2.metric("Total Profit", f"${total_profit:,.0f}")
        col3.metric("Total Orders", f"{total_orders:,}")
        col4.metric("Profit Margin", f"{profit_margin:.1f}%")

        st.markdown("---")

        # --- CHARTS ROW 1 ---
        c1, c2 = st.columns((1, 1))

        with c1:
            st.subheader("📈 Monthly Sales Trend")
            monthly_trend = df.set_index('order_date').resample('MS')['sales'].sum().reset_index()
            fig_trend = px.line(monthly_trend, x='order_date', y='sales', markers=True, 
                                 line_shape="spline")
            fig_trend.update_traces(line_color='#C5B358', line_width=3)
            st.plotly_chart(fig_trend, use_container_width=True)

        with c2:
            st.subheader("🏆 Sales by category")
            cat_sales = df.groupby('category')['sales'].sum().reset_index().sort_values('sales', ascending=False)
            fig_cat = px.bar(cat_sales, x='category', y='sales', color='sales', 
                              color_continuous_scale='magma')
            st.plotly_chart(fig_cat, use_container_width=True)

        # --- CHARTS ROW 2 ---
        c3, c4 = st.columns((1, 1))
        
        with c3:
            st.subheader("📦 Top Sub-Categories")
            sub_sales = df.groupby('sub_category')['sales'].sum().reset_index().sort_values('sales', ascending=False).head(10)
            fig_sub = px.bar(sub_sales, x='sales', y='sub_category', orientation='h', 
                              color='sales')
            fig_sub.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_sub, use_container_width=True)

        with c4:
            st.subheader("🗺️ Regional Distribution")
            reg_sales = df.groupby('region')['sales'].sum().reset_index()
            fig_pie = px.pie(reg_sales, values='sales', names='region', hole=0.4, 
                             template="plotly_white")
            st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# PAGE 2: AI FORECASTER
# ==========================================
elif page == "🔮 AI Forecaster":
    st.title("🔮 Intelligent Restocking Engine")
    st.markdown("Select a category below to generate a real-time forecast and inventory plan.")

    if df is not None:
        # Input Section
        col_sel, col_space = st.columns([1, 2])
        with col_sel:
            selected_cat = st.selectbox("Select Product category", df['category'].unique())

        # Run Forecasting Engine
        historical, forecast, rmse = run_forecast(selected_cat, df)
        
        # Calculate Advice (Next Month)
        future_forecast = forecast[forecast['ds'] > historical['ds'].max()].copy()
        next_month = future_forecast.iloc[0]
        
        pred_demand = next_month['yhat']
        safety_stock = rmse * SERVICE_LEVEL_Z
        rec_order = pred_demand + safety_stock

        st.markdown("---")

        # --- ADVICE CARD ---
        st.subheader(f"📋 Restocking Advice: {selected_cat}")
        
        kpi1, kpi2, kpi3 = st.columns(3)
        
        kpi1.metric(
            label="📉 Predicted Demand (Next Month)",
            value=f"${pred_demand:,.2f}",
            help="What the AI thinks you will sell."
        )
        
        kpi2.metric(
            label="🛡️ Safety Stock Buffer",
            value=f"${safety_stock:,.2f}",
            help="Extra stock to handle volatility (95% Service Level)."
        )
        
        kpi3.metric(
            label="🛒 RECOMMENDED ORDER",
            value=f"${rec_order:,.2f}",
            delta="Action Required",
            delta_color="normal"
        )
        
        st.success(f"💡 Strategy: Order **${rec_order:,.0f}** worth of {selected_cat} to meet demand with 95% confidence.")

        # --- INTERACTIVE PLOT ---
        st.markdown("### 🗓️ 12-Month Forecast View")
        
        fig = go.Figure()

        # Historical Data
        fig.add_trace(go.Scatter(
            x=historical['ds'], y=historical['y'],
            mode='lines+markers', name='Actual History',
            line=dict(color="#800000", width=1.5)
        ))

        # Forecast Line
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            mode='lines', name='AI Forecast',
            line=dict(color="#14C45D", width=2)
        ))

        # Confidence Interval (Upper/Lower)
        # fig.add_trace(go.Scatter(
        #     x=forecast['ds'], y=forecast['yhat_upper'],
        #     mode='lines', marker=dict(color="#049AFD"),
        #     line=dict(width=0), showlegend=False
        # ))
        # fig.add_trace(go.Scatter(
        #     x=forecast['ds'], y=forecast['yhat_lower'],
        #     marker=dict(color="#049AFD"),
        #     line=dict(width=0), mode='lines',
        #     fill='tonexty', fillcolor='rgba(0, 204, 150, 0.2)',
        #     name='Confidence Interval'
        # ))

        fig.update_layout(
            title=f"Sales Forecast for {selected_cat}",
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            hovermode="x unified",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # --- RAW DATA EXPANDER ---
        with st.expander("View Detailed Forecast Data"):
            st.dataframe(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(12))