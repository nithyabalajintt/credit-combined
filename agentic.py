
# Import agnodata Agent and sample tools.
from agno.agent import Agent
# from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools 
from agno.agent import Agent, RunResponse
# from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.embedder.azure_openai import AzureOpenAIEmbedder
from agno.models.azure import AzureOpenAI as AOI
from agno.document.chunking.recursive import RecursiveChunking
from pydantic import BaseModel, Field
import os
from datetime import datetime
from dotenv import load_dotenv
from agno.utils.pprint import pprint_run_response
from openai import AzureOpenAI
import json
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler  

import warnings
warnings.filterwarnings("ignore")

load_dotenv()  # Load env variables from .env file

#GPT-4o
model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
version_number = os.environ.get("API_VERSION_GA")

#ADA
embedding_model_name = os.environ.get("ADA_AZURE_OPENAI_MODEL_NAME")
embedding_endpoint_url = os.environ.get("ADA_AZURE_OPENAI_ENDPOINT")
embedding_api_key = os.environ.get("ADA_AZURE_OPENAI_API_KEY")
embedding_deployment_name = os.environ.get("ADA_AZURE_OPENAI_DEPLOYMENT")
# version_number = os.environ.get("API_VERSION_GA")


# Azure OpenAI
client = AzureOpenAI(
        azure_endpoint=endpoint_url,
        azure_deployment=deployment_name,
        api_key=api_key,
        api_version=version_number,
    )


embeddings = AzureOpenAIEmbedder(id=embedding_model_name, api_key = embedding_api_key, azure_deployment= embedding_deployment_name, azure_endpoint=embedding_endpoint_url, api_version="2024-10-21")


knowledge_base = TextKnowledgeBase(
    path="data/txt_files",
    vector_db=ChromaDb(collection="finance_docs", embedder = embeddings),
    chunking_strategy=RecursiveChunking(chunk_size = 1000, overlap = 100),
)
knowledge_base.load(recreate=False)


# ---------------------------------------------------------------------
# Finance Agent: Uses multiple tools to generate a risk narrative.
# ---------------------------------------------------------------------
risk_analysis_finance_agent = Agent(
    model=AOI(id=model_name,
    api_key=api_key,
    azure_endpoint=endpoint_url,
    azure_deployment=deployment_name
    ),
    description="You are an agent that drafts a credit risk narrative for a corporate loan credit note request for a company based on the context provided to you from the knowledge base and tool to understand the company's financial figures, sentiment, news and stock market data if any",
    tools=[YFinanceTools(key_financial_ratios = True, stock_fundamentals = True)], #add ml tool also
    # run_id=run_id,
    # user_id=user,
    knowledge=knowledge_base,
    instructions="""Based on the context provided to you, generate a comprehensive credit narrative addressing the users query.
Provide a well-structured credit narrative that integrates all relevant information from the context.
Include sections under the following headings:
1. financial performance
2. industry position
3.credit risks
4.recommendations

""",
    # add_context_instructions = ""
    # use_tools=True,
    show_tool_calls=True,
    # debug_mode=True,
    # markdown=True
)


# ---------------------------------------------------------------------
# Narrative Calculation Agent: Uses ML score and narrative to calculate application risk score.
# ---------------------------------------------------------------------

class ScoreStructure(BaseModel):
    score: float = Field(..., description="The score you will generate for the loan application based on the narrative and ml model score provided to you, must be in the range of 0-100")
 
def ml_model() -> None:
    """
    Use this function to get and print the application score, risk score,
    and a list of financial features for a given company.

    No Args
    """

    # --- 1. Feature Names ---
    FEATURE_NAMES = [
        'Net Income Continuous Operations', 'Inventory', 'Net Profit Margin %',
        'Return on Equity %', 'Return on Assets %', 'Current Ratio',
        'Quick Ratio', 'Asset Turnover Ratio', 'Debt Equity Ratio',
        'Debt To Asset Ratio', 'Interest Coverage Ratio',
        'Loan to Collateral Ratio', 'Credit Score'
    ]
    TARGET_NAME = "Application Score"

    # --- 2. Sample Data ---
    if company_name == "Delhivery_Limited":
        sample_data = {
            'Net Income Continuous Operations': [-2491860000],
            'Inventory': [164260000],
            'Net Profit Margin %': [-3.060674777],
            'Return on Equity %': [-2.724938724],
            'Return on Assets %': [-2.175723084],
            'Current Ratio': [4.418925633],
            'Quick Ratio': [4.406726807],
            'Asset Turnover Ratio': [0.710863859],
            'Debt Equity Ratio': [0.127871544],
            'Debt To Asset Ratio': [0.102098835],
            'Interest Coverage Ratio': [-1.785605215],
            'Loan to Collateral Ratio': [0.840480171],
            'Credit Score': [468]
        }
    else:
        print(f"❌ Data not available for company: {company_name}")
        return

    # --- 3. Load Model and Scaler ---
    try:
        with open('linear_regression_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError as e:
        print(f"❌ Error loading model or scaler: {e}")
        return

    # --- 4. Preprocess and Predict ---
    try:
        sample_df = pd.DataFrame(sample_data, columns=FEATURE_NAMES)
        input_features = sample_df
        scaled_features = scaler.transform(input_features)
        application_score = float(model.predict(scaled_features)[0])
        risk_score = 100 - application_score
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return

    # --- 5. Pretty Print Results ---
    print(f"\nCompany Name     : {company_name}")
    print(f"Features         : {FEATURE_NAMES}")
    print(f"Application Score: {round(application_score, 3)}")
    print(f"Risk Score       : {round(risk_score, 3)}")

#  # Example usage
# if __name__ == "__main__":
#     ml_model("Delhivery_Limited")

risk_calculation_narrative_agent = Agent(
    model=AOI(id=model_name,
    api_key=api_key,
    azure_endpoint=endpoint_url,
    azure_deployment=deployment_name
    ),
 
    description="You are an agent that calculates an application risk score based on a credit narrative and a Machine Learning model score and a list of features affecting that particular score.",
    tools=[ml_model], #add ml tool also
    # run_id=run_id,
    # user_id=user,
    # knowledge=knowledge_base,
    instructions="""Based on the narrative and the ML model score provided to you, generate your own application risk score which ranges between 0-100, where 0 would indicate no risk in approving a loan and 100 indicates the highest risk in approving the loan.""",
    # add_context_instructions = ""
    # use_tools=True,
    show_tool_calls=True,
    response_model=ScoreStructure,
    # debug_mode=True,
    # markdown=True
)

# ---------------------------------------------------------------------
# Feedback Agent: A simple agent that processes user feedback.
# (It does not require any external tools.)
# ---------------------------------------------------------------------
class FeedbackAgent:
    def process_feedback_narrative(self, feedback_text: str) -> str:
        feedback_chat_completion = client.chat.completions.create(

        messages=[
            {
                "role": "system",
                "content": f"""You are an expert feedback analyzer for credit narratives
    Analyze the feedback provided to you and structure it into clear improvement points to provide as instructions to another agent


    Structure the feedback into specific areas that need improvement, such as:
    1. Financial analysis issues
    2. Risk assessment gaps
    3. Missing information
    4. Structural problems
    5. Other specific areas for improvement

    For each area, provide clear and actionable suggestions."""
            },
            {
                "role": "user",
                "content": f"""FEEDBACK:
                {feedback_text}""",
            }
        ],

        model=model_name,
        # temperature=0.5,
        )
        return feedback_chat_completion.choices[0].message.content

    def process_feedback_note(self, feedback_text: str) -> str:
        feedback_chat_completion = client.chat.completions.create(

        messages=[
            {
                "role": "system",
                "content": f"""You are an expert feedback analyzer for credit notes
    Analyze the feedback provided to you and structure it into clear improvement points of missing information to provide as instructions to another agent


    Structure the feedback into specific areas that need improvement, such as:
    1. Note generation gaps
    2. Missing information
    3. Structural problems
    4. Other specific areas for improvement

    For each area, provide clear and actionable suggestions, if any"""
            },
            {
                "role": "user",
                "content": f"""FEEDBACK:
                {feedback_text}""",
            }
        ],

        model=model_name,
        # temperature=0.5,
        )
        return feedback_chat_completion.choices[0].message.content

feedback_agent = FeedbackAgent()

# ---------------------------------------------------------------------
# Credit Note Agent: Uses a provided template to generate a credit note.
# ---------------------------------------------------------------------
class CreditNoteAgent:
    def __init__(self, template: str):
        self.template = template

    def generate_credit_note(self, narrative: str, user_query:str, loan_details: dict,feedback_history: list = None) -> str:
        current_date = datetime.now().strftime("%Y-%m-%d")
        # Add feedback context to system prompt
        feedback_context = "\n".join([f"Feedback {i+1}: {fb}" for i, fb in enumerate(feedback_history or [])])
        system_prompt = f"""Generate credit note considering this feedback history:
        {feedback_context}
        
        {self.template}"""

        system_prompt = f"""You are a financial analyst generating formal credit notes. Use this template:
        {self.template}
        
        Replace these placeholders using:
        - [COMPANY_NAME]: {user_query}
        - [DATE]: {current_date}
        - [AMOUNT]: {loan_details['amount']}
        - [TERM]: {loan_details['term']} months
        - [PURPOSE]: {loan_details['purpose']}
        
        Fill other sections using the narrative below. Maintain professional tone and formatting."""

        credit_note_chat_completion = client.chat.completions.create(

        messages=[
            {
                "role": "system",
                "content": f"""Based on the credit narrative provided to you, generate a formal credit note using the provided template.
    Fill in all sections of the template with appropriate information from the narrative.

    TEMPLATE:
    {self.template}
    
    Replace the placeholder fields like [COMPANY_NAME], [DATE], etc. with actual content.
    For [DATE], use the current date.
    
    Make reasonable assumptions for any information not explicitly mentioned in the narrative,
    but ensure the credit note remains consistent with the narrative's assessment."""
            },
            {
                "role": "user",
                "content": f"""USER QUERY: {user_query}\n\nCREDIT NARRATIVE:
                {narrative}""",
            }
        ],

        model=model_name,
        temperature=0.4,
        )
        return credit_note_chat_completion.choices[0].message.content

#Adjust the template format as required.
credit_note_template = """
# CREDIT NOTE

## Company: [COMPANY_NAME]
## Date: [DATE]
## Credit Amount: [AMOUNT]
## Term: [TERM]

### Purpose of Loan
[PURPOSE]

### Financial Analysis
[FINANCIAL_ANALYSIS]

### Market Position
[MARKET_POSITION]

### Collateral & Security
[COLLATERAL]

### Covenants
[COVENANTS]

### Approval Conditions
[CONDITIONS]

### Approval Signatures
[SIGNATURES]
"""
credit_note_agent = CreditNoteAgent(credit_note_template)

# ---------------------------------------------------------------------
# Main pipeline: Combines all the agents and human-in-the-loop interactions.
# ---------------------------------------------------------------------
def main():
    input_data = "Generate me a credit note for a corporate loan for Microsoft"
    
    # Step 1: Generate and approve risk narrative
    # while True:  
    print("\n--- Generating Risk Narrative ---")
    narrative_response: RunResponse = risk_analysis_narrative_agent.run(input_data)
    pprint_run_response(narrative_response, markdown=True, show_time=True)
    data = fetch_data()
    score = rule_func(data) # Tree dt.py script
    score_response: RunResponse = risk_calculation_narrative_agent.run(narrative_response.content)
    pprint_run_response(score_response, markdown=True, show_time=True)

    while True:  # Approval inner loop
        approval = input("Do you approve the narrative? (yes/no): ").strip().lower()
        
        if approval == "yes":
            final_narrative_response = narrative_response.content  # Store final response
            break  # Exit inner loop

        # Feeback if user isn't happy
        user_feedback_narrative = input("Enter your feedback to improve the narrative: ")
        structured_fb_narrative = feedback_agent.process_feedback_narrative(user_feedback_narrative)
        input_data = input_data + " " + structured_fb_narrative 
        print("Feedback received. Regenerating narrative...\n")

        narrative_response: RunResponse = risk_analysis_narrative_agent.run(input_data)
        pprint_run_response(narrative_response, markdown=True, show_time=True)

        score_response: RunResponse = risk_calculation_narrative_agent.run(narrative_response.content)
        pprint_run_response(score_response, markdown=True, show_time=True)
        # print(score_response)

        # break  # Exit outer loop after approval

    # print(final_narrative_response)
    # Step 2: Generate and approve credit note
    # while True:  
    print("\n--- Generating Credit Note ---")
    credit_note = credit_note_agent.generate_credit_note(final_narrative_response, input_data)
    print("\nGenerated Credit Note:")
    print(credit_note)

    while True:  # Approval loop for credit note
        approval_cn = input("Do you approve the credit note? (yes/no): ").strip().lower()
        
        if approval_cn == "yes":
            break  # Exit credit note loop and complete the flow

        # Feedback if user isnt happy
        print("Feedback received. Regenerating credit note...\n")
        user_feedback_credit = input("Enter your feedback to improve the credit note: ")
        structured_fb_credit = feedback_agent.process_feedback_note(user_feedback_credit)
        input_data = input_data + " " + structured_fb_credit  # Update input with feedback
        # print(input_data)

        credit_note = credit_note_agent.generate_credit_note(final_narrative_response, input_data)
        # print("regen-note")
        print(credit_note)

        # break  #Exit outer loop after approval

    print("\nProcess Complete: Final narrative and credit note have been approved.")

if __name__ == "__main__":
    main()