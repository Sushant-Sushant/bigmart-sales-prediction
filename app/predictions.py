import pickle
import os

# Dynamically get absolute path to model/
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "bigmart_model.pkl")

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_sales(model, input_df):
    return model.predict(input_df)[0]
