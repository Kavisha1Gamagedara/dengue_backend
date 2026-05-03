import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import warnings

warnings.filterwarnings("ignore")

def train_save_model():
    print("Loading data...")
    df = pd.read_csv('../dengue_data_with_weather_data.csv')
    
    # Preprocessing: Fill missing values and sort by time
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    df = df.sort_values('Date')
    
    # Grouping by date (aggregate for a global model, or you could train per district)
    # For simplicity, we create a global trend model
    time_series = df.groupby('Date').agg({
        'Cases': 'sum',
        'Temp_avg': 'mean',
        'Precipitation_avg': 'mean',
        'Humidity_avg': 'mean'
    }).reset_index()
    
    time_series.set_index('Date', inplace=True)
    
    print("Training SARIMAX model...")
    # Target variable: Cases
    # Exogenous variables: Weather data
    exog = time_series[['Temp_avg', 'Precipitation_avg', 'Humidity_avg']]
    endog = time_series['Cases']
    
    # Basic SARIMAX parameters (p,d,q) x (P,D,Q,s)
    # In a real scenario, you'd use auto_arima to find these
    model = SARIMAX(endog, 
                    exog=exog,
                    order=(1, 1, 1), 
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    results = model.fit(disp=False)
    
    print("Saving model...")
    joblib.dump(results, 'sarimax_model.joblib')
    print("Model saved as sarimax_model.joblib")

if __name__ == "__main__":
    train_save_model()
