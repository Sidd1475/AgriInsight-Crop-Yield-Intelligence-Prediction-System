import joblib 
import pandas as pd 
from pathlib import Path

model_path = Path(__file__).parent.parent/"models"/"final_model.pkl"
model = joblib.load(model_path)




def predict_yield(input_data: dict):

    # Convert to DataFrame
    df = pd.DataFrame([input_data])


    # Year_end (since model expects this instead of Year)
    df["Year_end"] = df["Year"]

    # Drop Year because model doesn't expect it
    if "Year" in df.columns:
        df.drop(columns=["Year"], inplace=True)

    # Ensure exact feature order
    df = df[model.feature_names_in_]

    # Prediction
    prediction = model.predict(df)

    return float(prediction[0])