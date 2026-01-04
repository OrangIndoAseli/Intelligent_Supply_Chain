
# 📦 Intelligent Supply Chain Command Center

### *A Predictive Analytics & Inventory Optimization System*

## 📖 Project Overview
The **Intelligent Supply Chain Command Center** is an end-to-end Machine Learning application designed to solve the problem of **stockouts and overstocking** in retail. 

By leveraging historical sales data, this tool:
1.  **Forecasts Demand:** Uses Facebook Prophet to predict future sales for specific product categories.
2.  **Optimizes Inventory:** Calculates dynamic "Safety Stock" levels based on forecast uncertainty (RMSE) and a 95% Service Level.
3.  **Actionable Insights:** Outputs a clear "Restocking Plan" for store managers, reducing capital tied up in inventory while preventing lost revenue.

## 🚀 Key Features
* **Time-Series Forecasting:** automated monthly demand prediction using additive/multiplicative regression models.
* **Inventory Optimization Engine:** Dynamic safety stock calculation using statistical volatility metrics.
* **Interactive Dashboard:** A Streamlit-based UI for managers to view KPIs, trends, and generate live forecasts.
* **Category-Specific Tuning:** Custom hyperparameter tuning for distinct product lines (Furniture vs. Technology).

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Forecasting:** Prophet (Meta)
* **Web Framework:** Streamlit
* **Visualization:** Plotly, Matplotlib, Seaborn
* **Data Manipulation:** Pandas, NumPy

## 📂 Project Structure
```text
Intelligent_Supply_Chain/
│
├── data/
│   ├── raw/                   # Original sales data (CSV/Excel)
│   ├── processed/             # Cleaned data ready for modeling
│
├── notebooks/                 # Jupyter notebooks for experimentation
│   ├── 01_data_exploration.ipynb
│   ├── 02_forecasting_models.ipynb
│   └── ...
│
├── reports/                   # Generated forecasts and plots
│   ├── figures/               # Saved forecast visualization images
│   └── category_restocking_plan.csv
│
├── src/                       # Source code
│   ├── data_loader.py         # ETL pipeline to clean raw data
│   ├── forecast_by_category.py # Core ML Engine (Training & Inference)
│   └── app.py                 # Streamlit Dashboard application
│
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation

```

## ⚡ How to Run Locally

1. **Clone the repository:**
```bash
git clone [https://github.com/YOUR_USERNAME/Intelligent_Supply_Chain.git](https://github.com/YOUR_USERNAME/Intelligent_Supply_Chain.git)
cd Intelligent_Supply_Chain

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the ETL Pipeline (Data Cleaning):**
```bash
python src/data_loader.py

```


4. **Launch the Dashboard:**
```bash
streamlit run src/app.py

```



## 📊 Results & Business Impact

* **Model Accuracy:** Achieved a **RMSE reduction of ~15%** through hyperparameter tuning (Grid Search).
* **Cost Savings:** Identified potential monthly inventory savings of **~$2,400** by optimizing safety stock buffers.
* **Efficiency:** Reduced forecasting time from days (manual Excel) to seconds (automated AI).

## 🔮 Future Improvements

* Integration with live SQL databases for real-time inventory tracking.
* Deployment of the model as a REST API (FastAPI) for ERP integration.
* Adding "Sub-Category" level granularity for deeper insights.