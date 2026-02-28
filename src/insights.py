import pandas as pd 
import numpy as np 

def get_region_summary(df, district):

    region_df = df[df["District"]==district]

    if region_df.empty:
        return None
    summary = {}

    # Basic stats 
    summary["total_records"] = len(region_df)
    summary["avg_yield"] = round(region_df["Yield"].mean(),2)
    summary["total_production"] = round(region_df["Production"].median(),2)
    summary["total_area"] = round(region_df["Area"].sum(),2)

    # Diversity
    summary["crop_diversity"] = region_df["Crop"].nunique()
    summary["season_diversity"] = region_df["Season"].nunique()

    # Time range
    summary["first_year"] = region_df["Year_end"].min()
    summary["last_year"] = region_df["Year_end"].max()
    
    ## Growth calculation 
    yearly_mean = region_df.groupby("Year_end")["Yield"].mean()

    if len(yearly_mean) > 1:
        growth = ((yearly_mean.iloc[-1] - yearly_mean.iloc[0]) / yearly_mean.iloc[0]) *100
        summary["Yield_growth_percentage"] = round(growth,2)

    else:
        summary["Yield_growth_percentage"] = None
    
    return summary



def get_top_crops_by_yield(df, district):
    region_df = df[df["District"]==district]

    crop_yield = (region_df.groupby("Crop")["Yield"].mean().sort_values(ascending=False))
    top_5 = crop_yield.head(5)
    bottom_5 = crop_yield.tail(5)

    return {
        "top_5_crops":top_5.reset_index(),
        "bottom_5_crops": bottom_5.reset_index(),
        "best_crop": top_5.index[0],
        "worst_crop":bottom_5.index[-1],
        "crop_diversity": region_df["Crop"].nunique()
    }

def yearly_trend(df , district):

    region_df = df[df["District"]==district]

    yearly = region_df.groupby("Year_end")["Yield"].mean()
    growth = ((yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0]) *100 

    peak_year = yearly.idxmax()
    lowest_year = yearly.idxmin()

    volatility = yearly.std()

    return {
        "yearly_yield": yearly , 
        "growth_percent" : round(growth,2),
        "peak_year":peak_year,
        "lowest_year" : lowest_year ,
        "volatility": round(volatility,2)
    }

def features_importance_insights(model):

    regressor = model.named_steps["regressor"]
    preprocessor = model.named_steps["preprocessor"]

    features_name = preprocessor.get_feature_names_out()
    
    importances = regressor.feature_importances_

    importances_df = (
        pd.DataFrame({
            "Feature": features_name , 
            "Importance" : importances 
        }).sort_values(by="Importance",ascending=False)
    )

    return {
        "top_features": importances_df.head(10),
        "most_important_feature": importances_df.iloc[0]["Feature"],
        "num_features_used" : len(features_name)
    }



