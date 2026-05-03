import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json
import os
import datetime

# --- 1. INITIALIZE FIREBASE ---
# GitHub Actions will inject the FIREBASE_CREDENTIALS secret here
firebase_creds_json = os.environ.get('FIREBASE_CREDENTIALS')
creds_dict = json.loads(firebase_creds_json)
cred = credentials.Certificate(creds_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- 2. MACHINE LEARNING PIPELINE ---
# Load historical data
df = pd.read_csv('satta_disawar_historical_data_2022_2026.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# (Insert your data cleaning and ML feature engineering logic here)
# e.g., Month_Sin, Month_Cos, Lag_1, Is_Repeating, Days_Since_Last_Repeating

# Define features and target
features = ['Month_Sin', 'Month_Cos', 'DoW_Sin', 'DoW_Cos', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_1_Is_Repeating', 'Lag_2_Is_Repeating', 'Days_Since_Last_Repeating']
X = df.dropna()[features]
y = df.dropna()['Winning_Number'].astype(int)

# Train the Model
rf = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=8, min_samples_split=5)
rf.fit(X, y) # Add sample_weights here if implementing the MLOps loop

# Calculate Tomorrow's Target Date
last_row = df.iloc[-1]
next_date = last_row['Date'] + pd.Timedelta(days=1)
target_date_str = next_date.strftime('%Y-%m-%d')

# Predict
next_features = pd.DataFrame({
    'Month_Sin': [np.sin(2 * np.pi * next_date.month / 12)], 
    'Month_Cos': [np.cos(2 * np.pi * next_date.month / 12)],
    'DoW_Sin': [np.sin(2 * np.pi * next_date.dayofweek / 7)],
    'DoW_Cos': [np.cos(2 * np.pi * next_date.dayofweek / 7)],
    'Lag_1': [last_row['Winning_Number']],
    'Lag_2': [df.iloc[-2]['Winning_Number']],
    'Lag_3': [df.iloc[-3]['Winning_Number']],
    'Lag_1_Is_Repeating': [last_row['Is_Repeating']],
    'Lag_2_Is_Repeating': [df.iloc[-2]['Is_Repeating']],
    'Days_Since_Last_Repeating': [last_row['Days_Since_Last_Repeating'] + 1] # Simplified
})

probs = rf.predict_proba(next_features)[0]
classes = rf.classes_
prob_dict = {str(classes[i]): round(float(probs[i]) * 100, 2) for i in range(len(classes))}
sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

# --- 3. PUSH TO FIREBASE ---
# Structure the data document
prediction_data = {
    "target_date": next_date,
    "top_prediction": int(sorted_probs[0][0]),
    "top_probability_percent": sorted_probs[0][1],
    "runner_up_1": {"number": int(sorted_probs[1][0]), "probability": sorted_probs[1][1]},
    "runner_up_2": {"number": int(sorted_probs[2][0]), "probability": sorted_probs[2][1]},
    "runner_up_3": {"number": int(sorted_probs[3][0]), "probability": sorted_probs[3][1]},
    "timestamp": firestore.SERVER_TIMESTAMP
}

# Write to Firestore collection named 'daily_predictions'
doc_ref = db.collection('daily_predictions').document(target_date_str)
doc_ref.set(prediction_data)

print(f"Successfully pushed prediction for {target_date_str} to Firebase.")
