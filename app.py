import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import joblib
import pandas as pd
from pymongo import MongoClient
import certifi
import firebase_admin
from firebase_admin import credentials, messaging
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from google.oauth2 import id_token
from google.auth.transport import requests

app = Flask(__name__)
# Absolute permissive CORS for production
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept"]
}})
bcrypt = Bcrypt(app)

# --- Configuration ---
local_env = Path(__file__).resolve().parent / ".env"
parent_env = Path(__file__).resolve().parent.parent / ".env"

if local_env.exists():
    load_dotenv(local_env)
    print("Loaded .env from current directory")
elif parent_env.exists():
    load_dotenv(parent_env)
    print("Loaded .env from parent directory")
else:
    print("No .env file found. Using environment variables from system.")

MONGODB_URI = os.getenv("MONGODB_URI")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super-secret-key-dengue-2024")
GOOGLE_CLIENT_ID = "734514045592-m9p44jhei0h6i3ra723avjm1sburatkb.apps.googleusercontent.com" # Web Client ID for verification

if not MONGODB_URI:
    raise RuntimeError(
        "MONGODB_URI not set. Please set MONGODB_URI environment variable (or add it to a .env file)."
    )

app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=1)
jwt = JWTManager(app)

# --- MongoDB Connection ---
def get_db_connection():
    try:
        # Use certifi for SSL/TLS verification to avoid handshake errors on Windows/macOS/Linux
        ca = certifi.where()
        
        # Initialize MongoClient with robust settings for Atlas
        # tls=True is implied by mongodb+srv but we set it explicitly for clarity
        client = MongoClient(
            MONGODB_URI,
            tlsCAFile=ca,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            tls=True
        )
        
        # Test connection immediately by pinging the admin database
        # This will catch SSL/TLS errors or IP Whitelisting issues early
        client.admin.command('ping')
        
        # Determine database name: use the one in the URI if available, otherwise default
        db_name = MONGODB_URI.split("/")[-1].split("?")[0] or "Dengue_prediction_db"
        database = client.get_database(db_name)
        
        print(f"Connected to MongoDB Atlas: {db_name}")
        return client, database
    except Exception as e:
        error_msg = str(e)
        if "TLSV1_ALERT_INTERNAL_ERROR" in error_msg:
            print("\nCRITICAL: SSL Handshake Failed with 'TLSV1_INTERNAL_ERROR'.")
            print("This usually indicates your IP is NOT whitelisted in MongoDB Atlas.")
            print("Action: Go to Atlas -> Network Access -> Add IP Address -> 'Allow Access from Anywhere' (or add your current IP).\n")
        elif "ServerSelectionTimeoutError" in error_msg:
            print(f"\nCRITICAL: Could not connect to any MongoDB nodes. Timeout: {e}\n")
        else:
            print(f"\nMongoDB Connection Error: {e}\n")
        raise RuntimeError(f"Database connection failed. Please check your network and Atlas Whitelist.")

# Initialize DB
client, db = get_db_connection()
prediction_logs = db.prediction_logs
users_collection = db.users

# Prime the database with a startup event
try:
    db.system_events.insert_one({
        "event": "backend_started",
        "timestamp": datetime.now(timezone.utc)
    })
    print("Database primed with startup event")
except Exception as e:
    print(f"Warning: Failed to prime database: {e}")

# --- Load Models ---
try:
    model = joblib.load('dengue_model.pkl')
    print("Standard Model loaded successfully")
except Exception as e:
    print(f"Error loading standard model: {e}")
    model = None

try:
    sarimax_model = joblib.load('sarimax_model.joblib')
    print("SARIMAX Model loaded successfully")
except Exception as e:
    print(f"Error loading SARIMAX model: {e}")
    sarimax_model = None

# --- Firebase Initialization ---
# Path to your firebase-service-account.json
FIREBASE_CRED_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
if FIREBASE_CRED_PATH and os.path.exists(FIREBASE_CRED_PATH):
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred)
    print("Firebase initialized successfully")
else:
    print("Firebase credentials not found. FCM will be disabled.")

# --- Helpers ---
def serialize_doc(doc):
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

@app.before_request
def log_request_info():
    if request.method != 'OPTIONS':
        print(f"📡 {request.method} {request.path} from {request.remote_addr}")

@app.route('/', methods=['GET', 'POST', 'HEAD', 'OPTIONS'])
def index():
    return jsonify({
        'status': 'online',
        'message': 'Dengue Shield Backend is running!',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'build': '6:40 AM'
    }), 200

@app.route('/ping', methods=['GET', 'POST', 'HEAD', 'OPTIONS'])
def ping():
    return jsonify({'message': 'Backend is reachable!', 'build': '6:40 AM'}), 200

# --- Authentication Routes ---

@app.route('/register', methods=['POST'])
@app.route('/register/', methods=['POST'])
def register():
    print(f"Received registration request: {request.remote_addr} for {request.path}")
    try:
        data = request.get_json()
        if not data:
            return jsonify({'message': 'Missing JSON body'}), 400
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'message': 'Email and password are required'}), 400

        # Check if user already exists
        if users_collection.find_one({'email': email}):
            return jsonify({'message': 'User with this email already exists'}), 409

        # Hash the password
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Create user document
        user_doc = {
            'name': name,
            'email': email,
            'password': hashed_password,
            'created_at': datetime.now(timezone.utc),
            'is_new_user': True  # Flag for onboarding
        }
        
        result = users_collection.insert_one(user_doc)

        return jsonify({
            'message': 'User registered successfully',
            'user_id': str(result.inserted_id)
        }), 201
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'message': 'Internal server error during registration'}), 500

@app.route('/login', methods=['POST'])
@app.route('/login/', methods=['POST'])
def login():
    print(f"Received login request: {request.remote_addr} for {request.path}")
    try:
        data = request.get_json()
        if not data:
            return jsonify({'message': 'Missing JSON body'}), 400
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'message': 'Email and password are required'}), 400

        user = users_collection.find_one({'email': email})
        
        if user and bcrypt.check_password_hash(user['password'], password):
            # Create access token
            access_token = create_access_token(identity=str(user['_id']))
            return jsonify({
                'message': 'Login successful',
                'access_token': access_token,
                'user': {
                    'name': user.get('name'),
                    'email': user.get('email'),
                    'id': str(user['_id']),
                    'is_new_user': user.get('is_new_user', False)
                }
            }), 200
        else:
            return jsonify({'message': 'Invalid email or password'}), 401
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'message': 'Internal server error during login'}), 500

@app.route('/google-login', methods=['POST'])
def google_login():
    try:
        data = request.get_json()
        token = data.get('id_token')

        if not token:
            return jsonify({'message': 'ID Token is required'}), 400

        # Verify the token
        try:
            idinfo = id_token.verify_oauth2_token(token, requests.Request(), GOOGLE_CLIENT_ID)

            # ID token is valid. Get the user's Google ID from the 'sub' claim.
            email = idinfo['email']
            name = idinfo.get('name', 'Google User')

            # Check if user exists, if not create
            user = users_collection.find_one({'email': email})
            is_new_user = False

            if not user:
                user_doc = {
                    'name': name,
                    'email': email,
                    'created_at': datetime.now(timezone.utc),
                    'is_new_user': True,
                    'auth_provider': 'google'
                }
                result = users_collection.insert_one(user_doc)
                user = users_collection.find_one({'_id': result.inserted_id})
                is_new_user = True

            # Create access token
            access_token = create_access_token(identity=str(user['_id']))

            return jsonify({
                'message': 'Google login successful',
                'access_token': access_token,
                'user': {
                    'name': user.get('name'),
                    'email': user.get('email'),
                    'id': str(user['_id']),
                    'is_new_user': is_new_user
                }
            }), 200

        except ValueError:
            # Invalid token
            return jsonify({'message': 'Invalid Google ID token'}), 401

    except Exception as e:
        print(f"Google login error: {e}")
        return jsonify({'message': 'Internal server error during Google login'}), 500

# --- Profile Routes ---

@app.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    try:
        user_id = get_jwt_identity()
        from bson import ObjectId
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        
        if not user:
            return jsonify({'message': 'User not found'}), 404
            
        return jsonify({
            'name': user.get('name'),
            'email': user.get('email'),
            'is_new_user': user.get('is_new_user', False)
        }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    try:
        user_id = get_jwt_identity()
        from bson import ObjectId
        data = request.get_json()
        
        update_data = {}
        if 'name' in data:
            update_data['name'] = data['name']
        if 'is_new_user' in data:
            update_data['is_new_user'] = data['is_new_user']
            
        if not update_data:
            return jsonify({'message': 'No data provided to update'}), 400
            
        result = users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': update_data}
        )
        
        return jsonify({'message': 'Profile updated successfully'}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

# --- Prediction & Data Routes ---

@app.route('/predict', methods=['POST'])
@jwt_required()
def predict_risk():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
        
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        required_fields = ['Temp_avg', 'Precipitation_avg', 'Humidity_avg']
        for field in required_fields:
            if field not in data:
                return jsonify({'status': 'error', 'message': f'Missing field: {field}'}), 400

        # Create the exogenous DataFrame with features in the same order as training
        input_df = pd.DataFrame([{
            'Temp_avg': data.get('Temp_avg'),
            'Precipitation_avg': data.get('Precipitation_avg'),
            'Humidity_avg': data.get('Humidity_avg'),
            'Rainfall_Lag_1': data.get('Rainfall_Lag_1', data.get('Precipitation_avg'))
        }])
        
        # Statsmodels SARIMAX forecast for the next step using the provided weather data
        if hasattr(model, 'get_forecast'):
            forecast = model.get_forecast(steps=1, exog=input_df)
            predicted_cases = max(0.0, round(float(forecast.predicted_mean.iloc[0]), 2))
        elif hasattr(model, 'predict'):
            # Fallback for standard scikit-learn model
            pred = model.predict(input_df)
            predicted_cases = max(0.0, round(float(pred[0]), 2))
        else:
            return jsonify({'status': 'error', 'message': 'Model format unknown'}), 500

        risk_level = 'High' if predicted_cases > 100 else 'Low'

        log_entry = {
            "user_id": current_user_id,
            "temp": data.get('Temp_avg'),
            "precip": data.get('Precipitation_avg'),
            "humidity": data.get('Humidity_avg'),
            "predicted_cases": predicted_cases,
            "risk_level": risk_level,
            "timestamp": datetime.now(timezone.utc)
        }
        
        prediction_logs.insert_one(log_entry)

        return jsonify({
            'status': 'success',
            'predicted_cases': predicted_cases,
            'risk_level': risk_level
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/predict_sarimax', methods=['POST'])
@jwt_required()
def predict_sarimax():
    if sarimax_model is None:
        return jsonify({'status': 'error', 'message': 'SARIMAX model not loaded'}), 500
        
    try:
        data = request.get_json()
        # SARIMAX usually forecasts the next step based on historical context
        # But we can provide new exogenous variables for the forecast
        exog_data = pd.DataFrame([{
            'Temp_avg': data.get('Temp_avg'),
            'Precipitation_avg': data.get('Precipitation_avg'),
            'Humidity_avg': data.get('Humidity_avg')
        }])
        
        # Forecast 1 step ahead
        forecast = sarimax_model.get_forecast(steps=1, exog=exog_data)
        
        # Clip negative predictions to 0 for realistic output
        predicted_cases = max(0.0, round(float(forecast.predicted_mean.iloc[0]), 2))
        risk_level = 'High' if predicted_cases > 100 else 'Low'
        
        return jsonify({
            'status': 'success',
            'predicted_cases': predicted_cases,
            'risk_level': risk_level,
            'method': 'SARIMAX'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/heatmap', methods=['GET'])
def get_heatmap_data():
    try:
        csv_path = 'dengue_data_with_weather_data.csv'
        if not os.path.exists(csv_path):
            csv_path = '../dengue_data_with_weather_data.csv'

        df = pd.read_csv(csv_path)
        # Get the latest month data
        latest_data = df[df['Year'] == df['Year'].max()]
        latest_data = latest_data[latest_data['Month'] == latest_data['Month'].max()]
        
        import numpy as np
        max_cases_log = np.log1p(latest_data['Cases'].max())

        heatmap_points = []
        for _, row in latest_data.iterrows():
            # Use logarithmic scaling so smaller outbreaks are still visible
            # compared to extreme hotspots like Colombo
            weight = np.log1p(row['Cases']) / max_cases_log
            
            heatmap_points.append({
                'lat': row['Latitude'],
                'lng': row['Longitude'],
                'weight': weight,
                'district': row['District'],
                'cases': int(row['Cases'])
            })
            
        return jsonify(heatmap_points), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        csv_path = 'dengue_data_with_weather_data.csv'
        if not os.path.exists(csv_path):
            csv_path = '../dengue_data_with_weather_data.csv'

        df = pd.read_csv(csv_path)
        # Get the latest month data
        latest_year = df['Year'].max()
        latest_month_data = df[df['Year'] == latest_year]
        latest_month = latest_month_data['Month'].max()
        latest_data = latest_month_data[latest_month_data['Month'] == latest_month]
        
        total_cases = int(latest_data['Cases'].sum())
        # Mocking active cases as a fraction of total cases for the month
        # In a real scenario, this would come from a database of current cases
        active_cases = int(total_cases * 0.15) 
        recovered_cases = total_cases - active_cases
        
        # Risk area is the district with maximum cases
        risk_area_row = latest_data.loc[latest_data['Cases'].idxmax()]
        risk_area = risk_area_row['District']
        
        # Determine overall risk level
        if active_cases > 1000:
            risk_level = "High"
            risk_desc = "Significant increase in cases detected. Please take immediate precautions."
        elif active_cases > 300:
            risk_level = "Moderate"
            risk_desc = "Cases are present in your area. Maintain standard prevention measures."
        else:
            risk_level = "Low"
            risk_desc = "Case counts are currently low. Stay vigilant and keep environment clean."

        return jsonify({
            'total_cases': total_cases,
            'active_cases': active_cases,
            'recovered': recovered_cases,
            'risk_area': risk_area,
            'risk_level': risk_level,
            'risk_desc': risk_desc,
            'year': int(latest_year),
            'month': int(latest_month)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/register_fcm_token', methods=['POST'])
@jwt_required()
def register_fcm_token():
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        token = data.get('fcm_token')
        
        if not token:
            return jsonify({'message': 'Token is required'}), 400
            
        from bson import ObjectId
        users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {'fcm_token': token}}
        )
        return jsonify({'message': 'FCM Token registered successfully'}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/history', methods=['GET'])
@jwt_required()
def get_history():
    try:
        current_user_id = get_jwt_identity()
        # Fetch last 20 logs for this user
        logs = list(prediction_logs.find({'user_id': current_user_id}).sort('timestamp', -1).limit(20))
        for log in logs:
            log['_id'] = str(log['_id'])
            if isinstance(log['timestamp'], datetime):
                log['timestamp'] = log['timestamp'].isoformat()
        
        return jsonify(logs), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

if __name__ == '__main__':
    # Render provides the port in the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    print(f"Backend starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
