# AgriInsight — Crop Yield Prediction Dashboard

AgriInsight is a machine‑learning dashboard that predicts **crop yield per hectare** and helps you explore **regional and crop‑level patterns** — such as yield trends, top‑performing crops, and what’s driving the model’s predictions.  
It’s built with a preprocessing pipeline, a `RandomForestRegressor`, and a `Streamlit` interface that’s easy to demo and extend.

---

## 🚀 What you get in this repo

- The **full app**: `app.py`, `src/`, `notebooks/`, and preprocessed data in `data/preprocessed/`.
- A **reproducible training script**: run `python train_model.py` to rebuild the model locally.
- A **deployment‑ready model**: a compact `RandomForest` pipeline (~24.5 MB) saved as `models/final_model.pkl`.
- Earlier experimental models (including a very large ~1 GB version) are **not included**, because they were overfitted and not suitable for production.

---

<img width="1919" height="917" alt="image" src="https://github.com/user-attachments/assets/01586854-4a37-488b-8653-7f53c8f0e849" />

<img width="1901" height="944" alt="image" src="https://github.com/user-attachments/assets/afa9c106-906a-4911-ad68-956cd3c0df38" />

<img width="1900" height="916" alt="image" src="https://github.com/user-attachments/assets/178a3aa4-8a69-4c51-b46d-ff739a5febdb" />




---

## 🎯 What this app can do

- **Predict crop yield per hectare** for a given:
  - State, District, Crop, Season, Area, and Year.
- **Show district‑level insights**:
  - Average yield, total production, and year‑on‑year growth.
  - Yield trends over time.
  - Top and bottom‑performing crops.
  - Years of high or low yield and overall volatility.
- **Explain the model**:
  - Feature importance (which inputs matter most).
  - Simple‑language interpretations so farmers, extension officers, and policy makers can understand the “why” behind the numbers.

---

## 💡 Why this is useful

- Farmers and planners care more about **yield per hectare** (how much they get from each unit of land) than total production (which just reflects how much area is used).  
- By combining **predictions** with **regional insights**, users can:
  - Target support (training, inputs, credit) where it’s most needed.
  - Anticipate shortages or surpluses in supply.
  - Compare crops across districts and years.
- The **explanations** (feature importance and short text summaries) make the model more transparent and trustworthy for non‑technical audiences.

---

## 🏗️ How it works under the hood

- **Preprocessing** is handled by a `ColumnTransformer`:
  - Numeric features (`Area`, `Year_end`) are scaled with `StandardScaler`.
  - Categorical features (`Crop`, `Season`, `State`) are one‑hot encoded.
  - `District` is encoded using `TargetEncoder` to capture regional differences.
- **Model**:
  - `RandomForestRegressor` wrapped in an `sklearn Pipeline` that combines preprocessing and regression.
- **UI**:
  - Built with `Streamlit` (`app.py`) and split into three pages:
    - **Yield Prediction** — enter values and get a prediction.
    - **District Insights** — explore regional patterns.
    - **Model Insights** — see what’s driving the model.

---

<!-- INSERT PHOTO: simple architecture diagram (input → preprocessing → Random Forest → Streamlit UI) -->

---

## 📦 What model is actually used

The current production model is configured like this:

```python
RandomForestRegressor(
    n_estimators=150,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=12,      # limited to keep model small and stable
    random_state=42,
    n_jobs=-1
)
Setting max_depth=None produced extremely deep trees and a model that was around 1 GB — far too big and unstable for deployment.
By capping max_depth at 12, we get a compact model (~24.5 MB) that’s still accurate enough for most use cases.


📊 How well does it perform?
Earlier, “research‑style” model
In earlier experiments with a tuned RandomizedSearchCV and no tree depth limits:

Train R² ≈ 0.964

Test R² ≈ 0.88

These numbers look impressive, but:

The model was huge (~1 GB).

Some performance came from overfitting and a bit of data leakage.

It’s not suitable for deployment and is not included in this repo.

Current production model
The model you see here (final_model.pkl) is the production‑ready version:

Test R² ≈ 0.80

Model size ≈ 24.5 MB

Why the performance is a bit lower:

Constraining max_depth, min_samples_split, and other hyperparameters reduces model complexity and memory usage.

This improves generalization but usually lowers the R² slightly compared to an over‑fitted model.

For this project, stability, small size, and real‑world reliability are more important than squeezing out the last fraction of R².

⚠️ Limitations and things to watch out for
Very small areas:

Predictions for tiny plots (e.g., 0.1 ha) may be unreliable if the model rarely saw such examples.

The UI includes basic validation and warnings for extreme or unusual inputs.

Units:

Some datasets use quintals per hectare, others tonnes per hectare.

The app shows the unit you trained on; check data/preprocessed/cleaned_agridata.csv to confirm which one it is.

Regional data quality:

Reporting quality can vary between districts and states, which can bias the model’s outputs.

Be cautious when interpreting results for regions with sparse or inconsistent data.

Explainability:

We show feature importance and simple text explanations of key drivers.

SHAP was tried but is too heavy for live Streamlit; instead we use permutation importance and partial dependence plots (PDP) where helpful.

Model size:

Very deep trees lead to huge models that can’t load reliably in Streamlit.

That’s why we keep the Random Forest constrained in this version.

▶️ How to run this locally
Set up your environment (example):

bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
Install dependencies:

bash
pip install -r requirements.txt
Optional: retrain the model:

bash
python train_model.py
This script reads data/preprocessed/cleaned_agridata.csv and saves models/final_model.pkl.

Launch the app:

bash
streamlit run app.py
Check model size:

python
import os
print(os.path.getsize("models/final_model.pkl")/(1024*1024), "MB")

📁 Where to look in the code
app.py — main Streamlit app (entrypoint).

train_model.py — script that trains and saves models/final_model.pkl.

models/ — folder for trained models (the large experimental ones are not pushed).

data/preprocessed/cleaned_agridata.csv — the cleaned dataset used for training and experiments.

src/insights.py — functions for generating district‑ and crop‑level insights.

src/predict.py — prediction helpers called by the UI.

notebooks/ — exploratory notebooks for EDA and model experimentation (outputs cleared before committing).

🔁 Example: how to retrain the model
If you want to tweak the pipeline and retrain locally, here’s the core flow used in train_model.py:

python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from joblib import dump
from src.preprocessing import preprocessor
import pandas as pd

# Load data
df = pd.read_csv("data/preprocessed/cleaned_agridata.csv")
X = df.drop(columns=["Yield"])
y = df["Yield"]

# Define Random Forest
rf = RandomForestRegressor(
    n_estimators=150,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

# Build pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", rf)
])

# Train and save
pipeline.fit(X, y)
dump(pipeline, "models/final_model.pkl")
🔒 Keeping the repo size under control
We only save the final trained Pipeline, not the whole RandomizedSearchCV object or all cross‑validation results.

Tree depth and splitting rules are constrained to keep the model small.

The huge model (>100 MB) was removed from git history, and .gitignore rules prevent accidental large‑file commits.

<!-- INSERT PHOTO: small chart showing model size vs max_depth or number of trees -->
📣 What to show in demos or to reviewers
Short GIF or screenshots of:

The main Streamlit UI.

A sample prediction flow (inputs → output).

The District Insights and Model Insights pages.

A small “how to run” example:

Show train_model.py and streamlit run app.py so others can reproduce your results.

A brief note on units:

“Yield is in tonnes per hectare (or quintals per hectare) — confirmed from cleaned_agridata.csv.”

A clear trade‑off explanation:

“An unconstrained model gave R² ≈ 0.88 but was about 1 GB; the production model has R² ≈ 0.80 and is 24.5 MB, chosen for stability and generalization.”

➕ What could come next
A model registry (e.g., S3 or Hugging Face) so you don’t need to store models directly in GitHub.

Trying LightGBM or XGBoost with compression or quantization to get smaller models with similar performance.

Precomputing SHAP summaries offline and shipping only the visualizations (instead of calculating SHAP live in the app).

Adding prediction intervals so users can see when the model is less confident.

Improving the dataset:

Fixing unit mismatches (tonnes/ha vs quintals/ha),

Adding more small‑farm records,

Incorporating soil, fertilizer, and weather data.

📧 Who built this
Project author: Siddharth Mishra (repo owner)
