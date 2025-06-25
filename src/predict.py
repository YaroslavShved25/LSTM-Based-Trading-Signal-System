import numpy as np
import pickle
from tensorflow.keras.models import load_model
from src.config import MODEL_PATH, SCALER_PATH

def load_model_and_scaler():
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_next_price(model, scaler, input_data):
    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled.reshape(1, -1, input_data.shape[1]))
    return prediction[0][0]