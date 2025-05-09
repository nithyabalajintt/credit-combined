import pandas as pd
import requests
from rule_decision_new import rule_function  # You must define this function
 
EXCEL_FILE_PATH = "company_financial_fy2024.xlsx"
SHEET_NAME = "Financials"
 
def search_ticker_by_company_name(company_name):
    # try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={company_name}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        results = response.json().get("quotes", [])
        print(results)
 
        for result in results:
            symbol = result.get("symbol", "")
            # print(symbol)
            exchange = result.get("exchange", "")
            quote_type = result.get("quoteType", "")
            if symbol.endswith(".NS"):
                print(symbol)
            return symbol if symbol.endswith(".NS") else symbol - "BO" + ".NS"
 
        #  if quote_type in ["EQUITY", "ETF"] and exchange == "NS":
        #     return symbol if symbol.endswith(".NS") else symbol + ".NS"
        #  print(symbol)
 
        # return None
    # except Exception as e:
    #     print(f"Error fetching ticker for '{company_name}': {e}")
    #     return None
 
def safe_div(numerator, denominator):
    try:
        if denominator == 0 or denominator is None:
            return None
        return numerator / denominator
    except:
        return None
 
def fetch_financial_data_from_excel(ticker, loan_value, collateral_value, credit_score):
    # try:
        df = pd.read_excel("output\Company_Financials_FY2024.xlsx")
        # print(df)
        # pri
 
        if "Company" not in df.columns:
            print("Excel file must have a 'Ticker' column.")
            return None
 
        row = df[df['Company'] == ticker]
        if row.empty:
            print(f"Ticker {ticker} not found in Excel.")
            return None
 
        row = row.iloc[0]  # Get first matching row
        # print(row)
 
        data = {
            "Net Profit Margin %": row.get("Net Profit Margin %"),
            "Return on Equity %": row.get("Return on Equity %"),
            "Return on Assets %": row.get("Return on Assets %"),
            "Current Ratio": row.get("Current Ratio"),
            "Asset Turnover Ratio": row.get("Asset Turnover Ratio"),
            "Debt Equity Ratio": row.get("Debt Equity Ratio"),
            "Debt To Asset Ratio": row.get("Debt To Asset Ratio"),
            "Interest Coverage Ratio": row.get("Interest Coverage Ratio"),
            "Loan Value": loan_value,
            "Collateral Value": collateral_value,
            "Credit Score": credit_score,
        }
 
        data["LtC"] = safe_div(loan_value, collateral_value)
 
        return data
 
    # except Exception as e:
    #     print(f"Error reading Excel data: {e}")
    #     return None
 
def process_risk_result(company_name, ticker, risk_score, ltc, loan_value):
    print("\n--- Final Risk Evaluation ---")
    print(f"Company Name   : {company_name}")
    print(f"Ticker         : {ticker}")
    print(f"Loan Value     : {loan_value}")
    print(f"LtC (Loan/Collateral): {ltc}")
    print(f"Risk Score     : {risk_score}")
 
    # You can add logic to save this to a file or DB
 
def evaluate_company_risk(company_name, loan_value, collateral_value, credit_score):
    ticker = search_ticker_by_company_name(company_name)
    if not ticker:
        print(f"Ticker not found for company: {company_name}")
        return
 
    print(f"\nCompany: {company_name} | Ticker: {ticker}")
    data = fetch_financial_data_from_excel(ticker, loan_value, collateral_value, credit_score)
    if not data:
        return
 
    risk_score = rule_function(data)
    print(f"\nFinal Risk Score: {risk_score}")
 
    process_risk_result(company_name, ticker, risk_score, data.get("LtC"), loan_value)
 
if __name__ == "__main__":
    company = input("Enter company name: ").strip()
    loan = float(input("Enter loan value: "))
    collateral = float(input("Enter collateral value: "))
    credit = int(input("Enter credit score (300-900): "))
    
    evaluate_company_risk(company, loan, collateral, credit)