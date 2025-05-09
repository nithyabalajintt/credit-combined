import os
import pandas as pd

# Define the directory where Excel files are stored
folder_path = "Financial docs"

# Define the required features mapped to their respective sheets
required_features = {
    "Net Income Continuous Operations": "Income Statement",
    "Total Revenue": "Income Statement",
    "Stockholders Equity": "Balance Sheet",
    "Total Assets": "Balance Sheet",
    "Current Assets": "Balance Sheet",
    "Current Liabilities": "Balance Sheet",
    "Inventory": "Balance Sheet",
    "Total Debt": "Balance Sheet",
    "Interest Expense": "Income Statement",
    "EBIT": "Income Statement"
}

# Initialize an empty list to store processed data
processed_data = []

# Iterate over all Excel files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        file_path = os.path.join(folder_path, file_name)
        
        # Read the Excel file
        xls = pd.ExcelFile(file_path)
        
        # Initialize a dictionary to store extracted values
        data = {"Company Name": file_name.replace(".xlsx", "").replace(".xls", "")}
        
        for feature, sheet_name in required_features.items():
            if sheet_name in xls.sheet_names:  # Ensure the sheet exists
                df = xls.parse(sheet_name)
                
                # Extract the feature value if present in the first column
                if feature in df.iloc[:, 0].values:
                    value = df[df.iloc[:, 0] == feature].iloc[:, 1].values[0]
                    data[feature] = value
        
        # Compute the required ratios
        try:
            data["Net Profit Margin %"] = (data["Net Income Continuous Operations"] / data["Total Revenue"]) * 100
            data["Return on Equity%"] = (data["Net Income Continuous Operations"] / data["Stockholders Equity"]) * 100
            data["Return on Assets%"] = (data["Net Income Continuous Operations"] / data["Total Assets"]) * 100
            data["Current Ratio"] = data["Current Assets"] / data["Current Liabilities"]
            data["Quick Ratio"] = (data["Current Assets"] - data["Inventory"]) / data["Current Liabilities"]
            data["Asset Turnover Ratio"] = data["Total Revenue"] / data["Total Assets"]
            data["Interest Coverage Ratio"] = data["EBIT"] / data["Interest Expense"]
        except KeyError as e:
            print(f"Missing data in {file_name}: {e}")

        # Append the processed data
        processed_data.append(data)

# Convert processed data to DataFrame
final_df = pd.DataFrame(processed_data)

# Save the output
final_df.to_excel("final_output.xlsx", index=False)
print("Processing complete! Data saved to final_output.xlsx")
