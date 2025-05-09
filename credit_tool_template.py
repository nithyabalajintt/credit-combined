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
from dotenv import load_dotenv
from agno.utils.pprint import pprint_run_response
from openai import AzureOpenAI
import json
import pypandoc
import markdown
# from weasyprint import HTML
from docx import Document
from spire.doc import *
from spire.doc.common import *
import warnings
warnings.filterwarnings("ignore")

load_dotenv()  # Load env variables from .env file

model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
version_number = os.environ.get("API_VERSION_GA")

client = AzureOpenAI(
        azure_endpoint=endpoint_url,
        azure_deployment=deployment_name,
        api_key=api_key,
        api_version=version_number,
    )


class CreditNoteAgent:
    def __init__(self, template: str):
        self.template = template

    def generate_credit_note(self, narrative: str, user_query:str) -> str:
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


def convert_markdown_text_to_pdf(markdown_text, output_pdf):
    # Convert Markdown to HTML
    html_content = markdown.markdown(markdown_text)

    # Render HTML to PDF
    HTML(string=html_content).write_pdf(output_pdf)
    print(f"PDF created: {output_pdf}")

def convert_markdown_text_to_docx(markdown_filename, output_docx):
    # Split the Markdown string into lines
    
    # Create a Document object
    document = Document()

    # Load a Markdown file
    document.LoadFromFile(markdown_filename)

    # Save it as a docx file
    document.SaveToFile("output/ToWord.docx", FileFormat.Docx2016)

    # Dispose resources
    document.Dispose()
    print(f"DOCX created: {output_docx}")

def write_markdown_file(filename, content):
    """Writes content to a Markdown file.

    Args:
        filename: The name of the Markdown file to create or overwrite.
        content: A string containing the content to write, including Markdown syntax.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)



# markdown_text = """
# # Example Markdown
# This is a sample Markdown text.

# ## Features
# - Point 1
# - Point 2

# **Bold Text** and *Italic Text*

# [Link](https://example.com)
# """



credit_note_template_1 = """# Credit Note

**Company Name:** {{company_name}}  
**Date:** {{date}}  

**Ref:** Credit Loan for {{loan_reference}}

---

## Loan Details

| S.No | Field                                      | Description                                                                 |
|------|--------------------------------------------|-----------------------------------------------------------------------------|
| 1    | Borrowers                                  | {{borrowers}}                                                               |
| 2    | Amount                                     | {{amount}}                                                                  |
| 3    | Purpose                                    | {{purpose}}                                                                 |
| 4    | Drawdown                                   | {{drawdown}}                                                                |
| 5    | Term                                       | {{term}}                                                                    |
| 6    | Repayment                                  | {{repayment}}                                                               |
| 7    | Security                                   | {{security}}                                                                |
| 8    | Security Cover                             | {{security_cover}}                                                          |
| 9    | Interest Rate                              | {{interest_rate}}                                                           |
| 10   | Admin Fees                                 | {{admin_fees}}                                                              |
| 11   | Other Conditions                           | {{other_conditions}}                                                        |
| 12   | Representations, Warranties, Covenants     | {{representations_warranties_covenants}}                                   |
| 13   | Events of Default                          | {{events_of_default}}                                                       |
| 14   | Legal Costs                                | {{legal_costs}}                                                             |
| 15   | Law and Jurisdiction                       | {{law_and_jurisdiction}}                                                    |

---

## Signatures

**Authorized Signatory (Lender):**  
Name: {{signatory_1}}  
Designation: {{designation_1}}  
Date: {{date_1}}  
Signature: _____________________________  

<br>

**Authorized Signatory (Borrower):**  
Name: {{signatory_2}}  
Designation: {{designation_2}}  
Date: {{date_2}}  
Signature: _____________________________"""

credit_note_agent = CreditNoteAgent(credit_note_template)

final_narrative_response = """Credit Narrative for Punjab National Bank (PNB) 
Loan Request
Financial Performance Access to recent financial performance and stock data for Punjab National Bank was limited. However, the following highlights are relevant: - The last reported 52-week high was ₹57.95, and the 52-week low was ₹57.05. - There were no explicit figures on market capitalization, P/E ratio, or earnings per share for the preparation of this assessment. - Additional metrics such as profit margins, asset utilization, and net income growth couldn't be sourced at this time, implying the need for comprehensive financial disclosures to improve risk profiling.
Industry Position - Punjab National Bank operates in the financial services sector, categorized particularly under banking. It deals significantly in extending credit, operating deposits, and performing as a central institution in India's public banking domain. - PNB holds a reputable position as one of the largest public sector banks in India. - The industry's reliance on deposit-growth models and ongoing developments in India's financial inclusion and digitization of services highlight an opportunity for stability and potential growth.
Credit Risks Risks in approving the loan request for working capital of ₹10,00,000 with a 12-month tenure include: 1. Profitability Uncertainty: Due to the absence of recent earnings and profitability details, gauging repayment capability is challenging. 2. Macro-Economic Challenges: Being in a public banking sector, key risks include high exposure to corporate credit which can result in significant non-performing assets (NPAs) during economic downturns. 3. Market Volatility: The fluctuating 52-week stock price indicates potential market uncertainty or limited investor interest. 4. Regulatory Overhang: Public sector banks including PNB are subject to stringent regulatory oversight, impacting flexibility in credit policies. 5. Limited Financial Ratios: Critical ratios such as return on assets (ROA) and debt-equity ratio are unavailable, further increasing credit unpredictability.
Recommendations 1. Financial Disclosures: Request a detailed set of financial statements from PNB, including financial ratios, profit & loss statements, and asset quality data. 2. Collateral Requirements: Consider demanding collateral to secure the loan, given the uncertain financial profile. 3. Short-Term Monitoring: Recommend performance-based milestones or a mechanism to monitor utilization of funds on a quarterly basis. 4. Sectoral Analysis: Compare PNB’s figures to industry averages, particularly profitability metrics of other public sector banks, to better understand its standing. 5. Policy Guarantees: Propose a government guarantee or credit insurance as an additional safeguard.
In summary, while Punjab National Bank has a formidable presence in the public banking domain, the lack of real-time financial data adds a layer of risk to the credit approval process. Further due diligence and collateralization efforts are recommended to ensure the loan's successful deployment and recovery."""

input_data = "generate a credit note"
credit_note = credit_note_agent.generate_credit_note(final_narrative_response, input_data)
# credit_note = ""
write_markdown_file("credit_note.md", credit_note)

# Output file paths
# output_pdf = 'example.pdf'
output_docx = 'example.docx'

# Convert
# convert_markdown_text_to_pdf(credit_note, output_pdf)
convert_markdown_text_to_docx("credit_note.md", output_docx)