import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
 
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.preprocessing import LabelEncoder
# from scipy.stats import randint
from sklearn.preprocessing import MinMaxScaler


def rule_function(data):
    df1 = pd.read_excel("Company_Financials_Synthetic_First100.xlsx")
    # print(data)
    df2 = pd.DataFrame([data])
    df = pd.concat([df1,df2], axis = 0)
    
    print(df.columns) 

    df = df[["Net Profit Margin %","Return on Equity %", "Return on Assets %", 
            "Current Ratio", "Asset Turnover Ratio", "Debt Equity Ratio", "Debt To Asset Ratio", 
            "Interest Coverage Ratio","Loan Value", "Collateral Value", "Credit Score"]]

    df["LtC"] = df["Loan Value"]/df["Collateral Value"]

    df = df.drop(columns=["Loan Value", "Collateral Value"])
    # print(df)

    # Fill missing values for other columns with median
    df = df.fillna(df.median(numeric_only=True))

    # copy the data 

    df_min_max_scaled = df.copy()
    
    # # apply normalization techniques 

    # for column in df_min_max_scaled.columns: 

    # 	df_min_max_scaled[column] = (1 + (df_min_max_scaled[column] - df_min_max_scaled[column].min())*100) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())	
    
    # # view normalized data 

    # print(df_min_max_scaled)

    # df_min_max_scaled.to_csv("output\Scaled_Data.csv")
    
    #   Normalization
    scalar = MinMaxScaler()
    scaled_data = scalar.fit_transform(df_min_max_scaled)
    #print(scaled_data)
    scaled_df = pd.DataFrame(scaled_data, columns=df_min_max_scaled.columns)
    #print(scaled_df)
    scaled_df = (1 +(100*scaled_df))
    print(scaled_df)

    #scaled_df.to_csv("output\Scaled_Data2.csv")

    #Assigning of Weights for the features

    new_columns = ["Financial Risk Score","Repayment Risk Score"]
    for col in new_columns:
        scaled_df[col] = pd.NA
        

    dict_fin_weights = {"Net Profit Margin %": 0.25 ,"Return on Equity %":0.25 , "Return on Assets %":0.25, 
                    "Current Ratio":0.25, "Asset Turnover Ratio":0.1, "Debt Equity Ratio":0.1, "Debt To Asset Ratio":-0.2}
                    
    dict_repay_weights = {"Interest Coverage Ratio":0.20, "Credit Score":0.65,"LtC":0.15}

    dict_weights = dict_fin_weights|dict_repay_weights

    for feature, weight in dict_weights.items():
        scaled_df[feature] = scaled_df[feature]* weight


    scaled_df["Financial Risk Score"] = (100 - scaled_df[list(dict_fin_weights.keys())].sum(axis=1))
    scaled_df["Repayment Risk Score"] = (100 - scaled_df[list(dict_repay_weights.keys())].sum(axis=1))
    scaled_df["Final Risk Score"] = (scaled_df["Financial Risk Score"]*0.3)+(scaled_df["Repayment Risk Score"]*0.7)
    print(scaled_df)

    scaled_df.to_csv("scaled_data.csv")
    return scaled_df.iloc[-1,-1], df["LtC"].iloc[-1]
 