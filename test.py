import joblib
model = joblib.load("models/crop_yield.pkl")
print(model.feature_names_in_)