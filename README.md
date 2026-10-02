📈Rossmann Store Sales Forecasting App

🙌An end-to-end Machine Learning web application using XGBoost and Streamlit to predict daily store sales.

🌐Live Demo  :- https://sonamyadav-123-rossmann-sales-forecasting-app-bewyuo.streamlit.app/

📌Project Overview :-
Predicts store turnover based on promotions, competition distance, store types, and holiday calendars.

⚙️Key Features :-
1. Temporal Breakdown: Extracted Year, Month, Day, DayOfWeek, WeekOfYear, and IsWeekend features from date inputs.
2. Competition Metrics: Calculated active competition duration in months and imputed missing distance values using median strategy.
3. Categorical Encoding: Converted store categories and holiday types into numeric representations for XGBoost compatibility.
4. Target Scaling: Applied Log Transformation on sales data to normalize distribution and reduce outlier impact.
5. Store Logic: Automatically sets sales prediction to zero when a store is closed.

🛠️Tech Stack :-
- Python 3.13
- XGBoost
- Streamlit
- Pandas & NumPy
- Joblib


📁Project Structure :-
```text

├── app.py                      # Streamlit UI & Inference Logic
├── rossmann_xgb_model.pkl      # Saved XGBoost Model & Metadata
├── requirements.txt            # Project Dependencies
├── Rossmann_Sales_Model.ipynb  # Training & EDA Notebook
└── README.md                   # Project Documentation
