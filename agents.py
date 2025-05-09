import os
from dotenv import load_dotenv
import warnings
from pprint import pprint
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from IPython.display import Image, display


# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

# Import necessary libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langgraph.graph import END, START, StateGraph

# Setup vector database
# Load documents from the web
urls = [
    # "https://lilianweng.github.io/posts/2023-06-23-agent/",
    # "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-11m/",
]
docs = [WebBaseLoader(url).load() for url in urls]
doc_list = [item for sublist in docs for item in sublist]  # Flattening the list

# Splitting documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
docs_split = text_splitter.split_documents(doc_list)

# Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

# Create FAISS vector store
faiss_vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
# Insert embeddings into the vector store 
faiss_vector_store.add_documents(docs_split)
print(f"Inserted {len(docs_split)} documents into vector store.")

# Setup retriever
retriever = faiss_vector_store.as_retriever()

# Setup tools
duckducksearch_tool = DuckDuckGoSearchRun()
finance_tools = [YahooFinanceNewsTool()]

# Initialize LLM
llm = ChatGroq(groq_api_key=api_key, model="llama-3.3-70b-versatile")
# llm = "add llama"

# Set up finance agent for stock info
# finance_agent = initialize_agent(
#     finance_tools,
#     llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
# )

# Credit Note Template
CREDIT_NOTE_TEMPLATE = """
# CREDIT NOTE

## Company: [COMPANY_NAME]
## Date: [DATE]
## Credit Amount: [AMOUNT]
## Term: [TERM]

### Purpose of Loan
[PURPOSE]

### Credit Risk Assessment
[RISK_ASSESSMENT]

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

### Recommendations
[RECOMMENDATIONS]

### Approval Signatures
[SIGNATURES]
"""

# Helper functions
def extract_company_name(text: str) -> str:
    """Extract company name from the input text"""
    # This is a simple implementation - could be improved with NER
    common_indicators = ["for", "about", "on", "regarding"]
    
    for indicator in common_indicators:
        if indicator in text.lower():
            parts = text.lower().split(indicator)
            if len(parts) > 1:
                company = parts[1].strip().split()[0].capitalize()
                return company
    
    # Default fallback - extract potential company name
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() in ["company", "corporation", "inc", "corp"]:
            if i > 0:
                return words[i-1].capitalize()
    
    # If all else fails, look for capitalized words that might be company names
    for word in words:
        if word[0].isupper() and len(word) > 2 and word.lower() not in ["the", "for", "and"]:
            return word
            
    return "Company"  # Default fallback

def format_documents(docs: List[Document]) -> str:
    """Format retrieved documents into a readable string"""
    if not docs:
        return "No internal documents found."
    
    formatted = []
    for i, doc in enumerate(docs):
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        formatted.append(f"Document {i+1}:\n{content}\n")
    
    return "\n".join(formatted)

def format_financial_data(financial_data: Dict[str, Any]) -> str:
    """Format financial metrics data into a readable string"""
    if not financial_data:
        return "No relevant financial metrics found."
    
    formatted = []
    for ticker, data in financial_data.items():
        formatted.append(f"Financial data for {ticker}:\n{data}\n")
    
    return "\n".join(formatted)

# State management
class WorkflowState(dict):
    """Dictionary-based state for the workflow"""
    @classmethod
    def create(cls, query):
        """Create a new workflow state with the given query"""
        instance = cls({
            "query": query,
            "documents": [],
            "web_information": "",
            "financial_metrics": {},
            "narrative": None,
            "narrative_approved": False,
            "feedback": None,
            "credit_note": None,
            "credit_note_approved": False,
            "attempts": 0,
            "max_attempts": 3,
        })
        
        # Debug print to confirm state was created correctly
        print(f"Created state with query: {instance.get('query', 'NOT FOUND')}")
        return instance
    
    # Add dictionary-like access methods to ensure compatibility
    # def __getstate__(self):
    #     return dict(self)
        
    # def __setstate__(self, state):
    #     self.update(state)

# Define workflow nodes
def narrative_gen(state):
    """Generate a comprehensive credit narrative by querying multiple data sources"""
    print("\n---CREDIT NARRATIVE GENERATION---")
    print(state)
    
    # Debug print to see what's in the state
    print(f"State keys at start of narrative_gen: {state.keys()}")
    
    # Make a copy of the state to avoid modifying the original
    updated_state = state.copy()
    
    # Access query safely with a default value if not found
    query = updated_state.get("query", "")
    if not query:
        print("Warning: No query found in state")
        query = "Generate credit note for Apple"  # Fallback default
    
    narrative_data = {}
    
    # 1. Vector DB Retrieval - Get internal documents
    try:
        print("Retrieving internal documents...")
        vector_docs = retriever.invoke(query)
        narrative_data["internal_documents"] = vector_docs
    except Exception as e:
        print(f"Vector retrieval error: {e}")
        narrative_data["internal_documents"] = []
    
    # 2. Web Search - Get current market information
    try:
        print("Performing web search...")
        company_name = extract_company_name(query)
        search_query = f"credit analysis financial performance {company_name}"
        web_results = duckducksearch_tool.run(search_query)
        narrative_data["web_information"] = web_results
    except Exception as e:
        print(f"Web search error: {e}")
        narrative_data["web_information"] = "No web information retrieved."
    
    # 3. Stock Data - Get financial metrics when applicable
    try:
        print("Fetching financial data...")
        company_name = extract_company_name(query)
        stock_data = {}
        
        if company_name:
            try:
                # Using the agent executor to get comprehensive stock information
                stock_result = finance_agent.invoke({
                    "input": f"Get latest financial metrics and credit indicators for {company_name}"
                })
                stock_data[company_name] = stock_result["output"]
            except Exception as stock_error:
                print(f"Error retrieving stock data: {stock_error}")
                stock_data[company_name] = "No financial data available."
                
        narrative_data["financial_metrics"] = stock_data
    except Exception as e:
        print(f"Stock data retrieval error: {e}")
        narrative_data["financial_metrics"] = {}
    
    # Generate the comprehensive narrative by combining all sources
    print("Generating comprehensive credit narrative...")
    
    # Format the combined data for the narrative generation
    combined_context = f"""
    INTERNAL KNOWLEDGE BASE INFORMATION:
    {format_documents(narrative_data.get('internal_documents', []))}
    
    CURRENT MARKET INFORMATION:
    {narrative_data.get('web_information', 'No current market information available.')}
    
    FINANCIAL METRICS AND INDICATORS:
    {format_financial_data(narrative_data.get('financial_metrics', {}))}
    """
    
    # Use feedback if available
    feedback_prompt = ""
    if state.get("feedback"):
        feedback_prompt = f"""
        PREVIOUS FEEDBACK:
        {state.get('feedback')}
        
        Please address all the issues mentioned in the feedback and improve the narrative accordingly.
        """
    
    # Use the LLM to generate the final narrative
    narrative_prompt = f"""
    Based on the following information, generate a comprehensive credit narrative addressing this query:
    "{query}"
    
    INFORMATION SOURCES:
    {combined_context}
    
    {feedback_prompt}
    
    Provide a well-structured credit narrative that integrates all relevant information from the sources above.
    Include sections on financial performance, industry position, credit risks, and recommendations.
    """
    
    narrative = llm.invoke(narrative_prompt)
    # narrative = "sample narrative here"
    
    # Return enhanced state
    updated_state = state.copy()
    updated_state["documents"] = narrative_data.get('internal_documents', [])
    updated_state["web_information"] = narrative_data.get('web_information', "")
    updated_state["financial_metrics"] = narrative_data.get('financial_metrics', {})
    updated_state["narrative"] = narrative
    updated_state["attempts"] = state.get("attempts", 0) + 1
    
    return updated_state

def human_approve_narrative(state):
    """
    Human verification step for the narrative
    """
    print("\n---HUMAN VERIFICATION: NARRATIVE---")
    print("Please review the generated credit narrative:")
    print(state["narrative"])
    
    # UI
    # For now, we'll simulate it with an input
    approval = input("\nApprove narrative? (yes/no): ").strip().lower()
    
    if approval == "yes":
        state["narrative_approved"] = True
        return "generate_credit_note"
    else:
        feedback = input("Please provide feedback for improvement: ")
        state["feedback"] = feedback
        return "process_feedback"

def process_feedback(state):
    """
    Process and structure feedback for improving the narrative
    """
    print("\n---PROCESSING FEEDBACK---")
    feedback = state["feedback"]
    
    # Use LLM to structure the feedback
    structured_feedback_prompt = f"""
    You are an expert feedback analyzer for credit narratives.
    Please analyze the following feedback and structure it into clear improvement points:

    FEEDBACK:
    {feedback}

    Structure the feedback into specific areas that need improvement, such as:
    1. Financial analysis issues
    2. Risk assessment gaps
    3. Missing information
    4. Structural problems
    5. Other specific areas for improvement

    For each area, provide clear and actionable suggestions.
    """
    
    structured_feedback = llm.invoke(structured_feedback_prompt)
    state["feedback"] = structured_feedback.content
    
    # Check if we've hit the maximum attempts
    if state["attempts"] >= state["max_attempts"]:
        print("Maximum revision attempts reached.")
        return "end_workflow"
    else:
        return "generate_narrative"

def generate_credit_note(state):
    """
    Generate a credit note based on the approved narrative
    """
    print("\n---GENERATING CREDIT NOTE---")
    
    company_name = extract_company_name(state["query"])  # Fixed: use "query" instead of "question"
    
    credit_note_prompt = f"""
    Based on the approved credit narrative below, generate a formal credit note using the provided template.
    Fill in all sections of the template with appropriate information from the narrative.
    
    CREDIT NARRATIVE:
    {state["narrative"]}
    
    TEMPLATE:
    {CREDIT_NOTE_TEMPLATE}
    
    Replace the placeholder fields like [COMPANY_NAME], [DATE], etc. with actual content.
    For [COMPANY_NAME], use: {company_name}
    For [DATE], use the current date.
    
    Make reasonable assumptions for any information not explicitly mentioned in the narrative,
    but ensure the credit note remains consistent with the narrative's assessment.
    """
    
    credit_note = llm.invoke(credit_note_prompt)
    # credit_note = "sample credit note here"
    
    state["credit_note"] = credit_note
    return state

def human_approve_credit_note(state):
    """
    Human verification step for the credit note
    """
    print("\n---HUMAN VERIFICATION: CREDIT NOTE---")
    print("Please review the generated credit note:")
    print(state["credit_note"])
    
    # In a real application, this would be a UI element
    approval = input("\nApprove credit note? (yes/no): ").strip().lower()
    
    if approval == "yes":
        state["credit_note_approved"] = True
        return "end_workflow"
    else:
        feedback = input("Please provide feedback for the credit note: ")
        state["feedback"] = feedback
        state["narrative_approved"] = False  # Reset approval to regenerate narrative
        return "process_feedback"

# Define workflow graph
def create_workflow():
    # Create state graph
    # workflow = StateGraph(WorkflowState)
    workflow = StateGraph(dict)
    
    
    # Add nodes with explicit state handling
    workflow.add_node("generate_narrative", narrative_gen)
    workflow.add_node("process_feedback", process_feedback)
    workflow.add_node("generate_credit_note", generate_credit_note)
    
    # Add conditional edges with proper state passing
    workflow.add_edge(START, "generate_narrative")
    workflow.add_conditional_edges(
        "generate_narrative",
        human_approve_narrative,
        {
            "generate_credit_note": "generate_credit_note",
            "process_feedback": "process_feedback"
        }
    )
    workflow.add_edge("process_feedback", "generate_narrative")
    workflow.add_conditional_edges(
        "generate_credit_note",
        human_approve_credit_note,
        {
            "end_workflow": END,
            "process_feedback": "process_feedback"
        }
    )
    
    
    
    # Compile the graph with config to preserve state
    return workflow.compile()

# Main execution function
def run_credit_workflow(query: str):
    """Run the complete credit workflow with the given query"""
    # Create workflow
    app = create_workflow()
    # try:
    #     display(Image(app.get_graph().draw_mermaid_png()))

    # except Exception:
    # # This requires some extra dependencies and is optional pass
    #     print("Error")
    #     pass
    
    # Create initial state
    input_state = WorkflowState.create(query)
    
    # Debug the initial state
    print(f"Initial state: {input_state}")
    print(f"Query in initial state: {input_state.get('query', 'NOT FOUND')}")
    
    # Execute workflow
    print(f"\nStarting credit workflow for query: {query}")
    print("\n" + "="*50)
    
    try:
        # Run the workflow with explicit state preservation
        current_state = input_state
        print("input state")
        print(current_state)
        for output in app.stream(current_state):
            print("NODE 1")
            print(output)
            for key, value in output.items():
                print(f"\nCompleted node: '{key}'")
                print(f"State keys after node execution: {value.keys() if value else 'None'}")
                print(f"DEBUG: State after '{key}' -> {value}")
                current_state = value
        
        # Final state is the last state we saw
        final_state = current_state
        
        # Print final
        print("\n" + "="*50)
        print("WORKFLOW COMPLETED")
        print("="*50)
        
        # if final_state.get("narrative_approved") and final_state.get("credit_note_approved"):
        #     print("\nBOTH NARRATIVE AND CREDIT NOTE APPROVED")
        # elif final_state.get("narrative_approved"):
        #     print("\nNARRATIVE APPROVED BUT CREDIT NOTE REJECTED")
        # else:
        #     print("\nNARRATIVE REJECTED")
        
        return final_state
    
    except Exception as e:
        print(f"Error in workflow execution: {e}")
        import traceback
        traceback.print_exc()  # Print the full stack trace
        return None


if __name__ == "__main__":
    
    input_data = "Generate me a credit note for a corporate loan for Apple"
    # query = "Generate me a credit note for a corporate loan for Apple"
    final_state = run_credit_workflow(input_data)