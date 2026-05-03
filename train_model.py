import pandas as pd
import numpy as np
import statsmodels.api as sm
import joblib
import warnings
from sklearn.metrics import mean_absolute_error
import os

warnings.filterwarnings("ignore")

def train_save_model():
    print("Loading and Preprocessing data...")
    # Adjust path if running from backend directory
    csv_path = '../dengue_data_with_weather_data.csv'
    if not os.path.exists(csv_path):
        csv_path = 'dengue_data_with_weather_data.csv' # Fallback for root execution
        
    df = pd.read_csv(csv_path)
    
    # 1. Forward fill missing weather data
    df['Temp_avg'] = df['Temp_avg'].ffill()
    df['Precipitation_avg'] = df['Precipitation_avg'].ffill()
    df['Humidity_avg'] = df['Humidity_avg'].ffill()

    # 2. CREATE THE CRUCIAL BIOLOGICAL FEATURE
    # Calculate previous month's rainfall for each specific district
    df['Rainfall_Lag_1'] = df.groupby('District')['Precipitation_avg'].shift(1)
    df = df.dropna() # Drop the first month since it won't have a lag
    
    # Sort chronologically
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    df = df.sort_values('Date')

    print("Setting up District-Aware SARIMAX training...")
    
    # Target and Exogenous Variables
    X = df[['Temp_avg', 'Precipitation_avg', 'Humidity_avg', 'Rainfall_Lag_1']]
    y = df['Cases']
    
    # Chronological Split (80/20) - No shuffling to preserve time-series!
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training SARIMAX (1,1,1)x(1,1,1,12) on {len(X_train)} samples...")
    
    # 3. Train the model
    model = sm.tsa.SARIMAX(
        endog=y_train, 
        exog=X_train,
        order=(1, 1, 1), 
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    results = model.fit(disp=False)
    
    # 4. Evaluate the model
    predictions = results.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=X_test)
    error = mean_absolute_error(y_test, predictions)
    print(f"Model successfully trained! Mean Absolute Error: {error:.2f} cases")
    
    print("Saving model...")
    # Saving as dengue_model.pkl so your app.py can find it!
    joblib.dump(results, 'dengue_model.pkl')
    print("Model saved successfully as dengue_model.pkl!")

if __name__ == "__main__":
    train_save_model()
