import pandas as pd
import boto3
import io
import joblib

BUCKET_NAME = "ev-charging-trevor-2026-679930074560-us-east-2-an"
MODEL_KEY = "models/range_anxiety_rf_model.pkl"

print("Connecting to AWS S3 from local machine...")
s3_client = boto3.client('s3')

# Download the live model artifact straight from S3 memory
print("Fetching your trained cloud model...")
model_response = s3_client.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
loaded_model = joblib.load(io.BytesIO(model_response['Body'].read()))

# DEFINE A TEST DRIVER PROFILE

test_driver = pd.DataFrame([{
    'Distance Driven (since last charge) (km)': 195.0,
    'Temperature (°C)': 5.0,
    'User Type_Casual Driver': 0,
    'User Type_Commuter': 0,
    'User Type_Long-Distance Traveler': 1
}])
# Predict the risk for this test driver using the loaded model
prediction = loaded_model.predict(test_driver)

print("\n--- LOCAL INFERENCE TEST ---")
if prediction[0] == 1:
    print("ALERT: High Risk of Range Anxiety!")
else:
    print("SAFE: Buffer is sufficient.")