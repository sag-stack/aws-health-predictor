import joblib
import os
import numpy as np
import json
import pandas as pd

def model_fn(model_dir):
    """
    Load the trained model from the SageMaker model directory.
    """
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load the model using joblib
    model = joblib.load(model_path)
    return model



def input_fn(input_data, content_type):
    """Parse the input data payload."""
    if content_type == 'application/json':
        # Parse JSON and convert to pandas DataFrame
        data = json.loads(input_data)
        return pd.DataFrame([data])  # Convert to a DataFrame with one row
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """
    Use the model to make a prediction on the input data.
    """
    prediction = model.predict(input_data)
    return prediction

def output_fn(prediction, accept):
    """
    Convert the prediction result into a desired response format.
    """
    if accept == 'application/json':
        response = json.dumps({'predictions': prediction.tolist()})
        return response
    else:
        raise ValueError(f"Accept type {accept} not supported")
