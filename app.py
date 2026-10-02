import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title='Rossmann Sales Predictor', page_icon='📈', layout='wide'
)


# Load Trained Model
@st.cache_resource
def load_model():
  return joblib.load('rossmann_xgb_model.pkl')


model_data = load_model()
model = model_data['model']
feature_cols = model_data['feature_cols']

# Application Header
st.title('📈 Rossmann Store Sales Forecasting App')
st.markdown(
    'Predict store sales using machine learning based on store features,'
    ' promotions, and temporal data.'
)
st.divider()

# Input Form
col1, col2, col3 = st.columns(3)

with col1:
  st.subheader('🏪 Store Details')
  store_id = st.number_input('Store ID', min_value=1, max_value=1115, value=1)
  store_type = st.selectbox(
      'Store Type', options=[1, 2, 3, 4], format_func=lambda x: f'Type {x}'
  )
  assortment = st.selectbox(
      'Assortment Level',
      options=[1, 2, 3],
      format_func=lambda x: f'Level {x}',
  )
  competition_dist = st.number_input(
      'Competition Distance (meters)', min_value=0, value=1270
  )

with col2:
  st.subheader('📅 Date & Calendar')
  selected_date = st.date_input('Select Date', value=datetime.date.today())
  is_open = st.radio('Is Store Open?', options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
  state_holiday = st.selectbox(
      'State Holiday',
      options=[0, 1, 2, 3],
      format_func=lambda x: [
          'None',
          'Public Holiday',
          'Easter',
          'Christmas',
      ][x],
  )
  school_holiday = st.selectbox(
      'School Holiday',
      options=[0, 1],
      format_func=lambda x: 'Yes' if x == 1 else 'No',
  )

with col3:
  st.subheader('📣 Promotions & Competition')
  promo = st.radio(
      'Is Store Running Promo Today?',
      options=[1, 0],
      format_func=lambda x: 'Yes' if x == 1 else 'No',
  )
  promo2 = st.radio(
      'Is Store Part of Continuous Promo2?',
      options=[1, 0],
      format_func=lambda x: 'Yes' if x == 1 else 'No',
  )
  comp_open_months = st.number_input(
      'Competition Open Duration (Months)', min_value=0, value=24
  )

# Preprocessing Input Data
date_obj = pd.to_datetime(selected_date)
year = date_obj.year
month = date_obj.month
day = date_obj.day
day_of_week = date_obj.dayofweek
week_of_year = int(date_obj.isocalendar().week)
is_weekend = 1 if day_of_week >= 5 else 0

input_dict = {
    'Store': store_id,
    'DayOfWeek': day_of_week,
    'Promo': promo,
    'StateHoliday': state_holiday,
    'SchoolHoliday': school_holiday,
    'StoreType': store_type,
    'Assortment': assortment,
    'CompetitionDistance': competition_dist,
    'CompetitionOpenMonths': comp_open_months,
    'Promo2': promo2,
    'Year': year,
    'Month': month,
    'Day': day,
    'WeekOfYear': week_of_year,
    'IsWeekend': is_weekend,
}

input_df = pd.DataFrame([input_dict])[feature_cols]

st.divider()

# Prediction Output
if st.button('🚀 Predict Sales Turnover', use_container_width=True):
  if is_open == 0:
    st.warning('⚠️ Store is CLOSED on this day. Estimated Sales: $0.00')
  else:
    log_pred = model.predict(input_df)[0]
    sales_pred = np.expm1(log_pred)
    st.success(
        f'💵 Estimated Sales Turnover: **${sales_pred:,.2f}**', icon='🎉'
    )