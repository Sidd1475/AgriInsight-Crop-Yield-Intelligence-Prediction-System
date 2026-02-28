import pandas as pd 
import joblib 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from category_encoders.target_encoder import TargetEncoder 

def train_model(data_path):

    df = pd.read_csv(data_path)

    df = df.drop(columns="Production Units")
    df = df.drop(columns="Production")
    df = df.drop(columns="Unnamed: 0")

    ## Year based Split 
    train = df[(df['Year_end']>=1998) & (df['Year_end']<= 2017)]
    test = df[(df['Year_end']>=2018) & (df['Year_end']<= 2020)]

    X_train = train.drop('Yield', axis=1)
    y_train = train['Yield']

    X_test = test.drop('Yield', axis=1)
    y_test = test['Yield']

    ## Removing the production and production units column column 



    ## Log Transformation 
    df['Area'] = np.log1p(df['Area'])
    df['Yield'] = np.log1p(df['Yield'])


    

    ## Column groups 
    numeric_cols = ['Area','Year_end']
    ohe_cols = ['Crop','Season','State']
    target_enc_cols = ['District']

    ## Build preprocessor 
    preprocessor = ColumnTransformer(
        transformers=[
            ('num',StandardScaler(),numeric_cols),
            ('ohe',OneHotEncoder(handle_unknown='ignore'),ohe_cols),
            ('target',TargetEncoder(),target_enc_cols)
        ]
    )

    ## final model 
    model = Pipeline([
        ('preprocessor',preprocessor),
        ('regressor',RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
        ))
    ])
    
    ## Train 

    model.fit(X_train,y_train)

    # Evaluate 
    test_r2 = model.score(X_test,y_test)
    print("Test R2:", test_r2)

    # Save Model 
    joblib.dump(model,"models/final_crop_model.pkl")

