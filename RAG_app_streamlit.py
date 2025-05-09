import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from openai import AzureOpenAI
import os

model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME")
endpoint_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
version_number = os.environ.get("API_VERSION_GA")

embedding_model_name = os.environ.get("ADA_AZURE_OPENAI_MODEL_NAME")
embedding_endpoint_url = os.environ.get("ADA_AZURE_OPENAI_ENDPOINT")
embedding_api_key = os.environ.get("ADA_AZURE_OPENAI_API_KEY")
embedding_deployment_name = os.environ.get("ADA_AZURE_OPENAI_DEPLOYMENT")

# Function to process uploaded document and initialize Chroma database
def initialize_chroma(uploaded_file):
    print(uploaded_file)
    with open(f"temp_data/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())


    # Load document using PyPDFLoader
    # if uploaded_file is not None:
    #     documents = [uploaded_file.read().decode()]
    loader = PyPDFLoader(f"temp_data/{uploaded_file.name}")
    documents = loader.load()

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Embed documents into Chroma database
    embedder = AzureOpenAIEmbeddings(model=embedding_model_name, api_key = embedding_api_key, azure_endpoint=embedding_endpoint_url, api_version="2024-10-21")
    db = Chroma.from_documents(texts, embedder, persist_directory="temp_chroma_db")
    db.persist()
    return db

# Application logic
def query_rag(db, query_text):
    # Retrieve the context from the DB using similarity search
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # Combine context from matching documents
    if len(results) == 0 or results[0][1] < 0.7:
        return "I'm sorry, I couldn't find any relevant information for your query."

    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _ in results])

    # Call the Azure OpenAI model
    client = AzureOpenAI(
            azure_endpoint=endpoint_url,
            azure_deployment=deployment_name,
            api_key=api_key,
            api_version=version_number,
        )

    rag_chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Retrieval Augmentation Chatbot that will only reply to queries "
                    "asked by the user based on the context provided to you from the vector database. "
                    "If the answer to the query does not lie in the context provided to you, you will simply reply "
                    'with "I am sorry, that question lies beyond the context provided to me, I cannot answer it."'
                )
            },
            {
                "role": "user",
                "content": f"VECTOR DATABASE CONTEXT:\n{context_text}\n\nUSER QUERY:{query_text}",
            },
        ],
        model=model_name,  # Specify the model
    )

    # Generate response text
    response_text = rag_chat_completion.choices[0].message.content

    # Get sources of the matching documents
    sources = [doc.metadata.get("source", None) for doc, _ in results]

    # Format response
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response

# Streamlit app
def main():
    st.title("Retrieval-Augmented Generation (RAG) Query App")
    st.write(
        "Upload a document, and then you can query it using the power of Retrieval-Augmented Generation (RAG)."
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], accept_multiple_files=False)

    if uploaded_file is not None:
        if "db" not in st.session_state:
            st.success("File uploaded successfully!")
            
            # Initialize the Chroma database and store it in session state
            with st.spinner("Processing the document..."):
                st.session_state.db = initialize_chroma(uploaded_file)

            st.write("Document indexed successfully! You can now query it.")
        else:
            st.write("Document has already been indexed. You can now query it.")

        # Query input
        query_text = st.text_input("Enter your query:")

        if st.button("Submit Query"):
            if query_text.strip():
                with st.spinner("Getting your response..."):
                    response = query_rag(st.session_state.db, query_text)
                st.write("### Response:")
                st.write(response)
            else:
                st.warning("Please enter a query before submitting!")
if __name__ == "__main__":
    main()