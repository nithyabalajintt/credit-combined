import os
from create_eval_dataset import create_eval_ds
from agno.agent import Agent
from agno.document.chunking.fixed import FixedSizeChunking
from agno.embedder.ollama import OllamaEmbedder
from agno.models.anthropic import Claude
# from qdrant_client import qdrant_client
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.qdrant import Qdrant
from dotenv import load_dotenv, find_dotenv

from ragas.llms import LlamaIndexLLMWrapper #USe Azure openai
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, FactualCorrectness, ContextRelevance, ContextUtilization, ContextRecall

# from llama_index.llms.openai import OpenAI

eval_llm = OpenAI(model='gpt-4o') # replace with azure openai

load_dotenv(find_dotenv())

doc_path = "data/test_data.pdf" # define test data
ground_truth_path = "data/ground_truth.json" # define ground truth data 
chunk_size = 1000
chunk_overlap = 200

# initialize the LLM (default to openai)
# claude = Claude(id="claude-3-7-sonnet-20250219")

# initialize the qdrant client.
# replace here with chroma instead of qdrant
q_client = qdrant_client.QdrantClient(url=os.environ.get('qdrant_url'), api_key=os.environ.get('api_key'))

# create the chroma vector store instance
vector_db = Qdrant(
    collection=os.environ.get('collection_name'),
    url=os.environ.get('qdrant_url'),
    api_key=os.environ.get('api_key'),
    embedder=OllamaEmbedder(id="nomic-embed-text:latest", dimensions=768)
)

# configure the knowledge base
knowledge_base = PDFKnowledgeBase(vector_db=vector_db,
                                  path=doc_path,
                                  chunking_strategy=FixedSizeChunking(  # try different methods here
                                      chunk_size=chunk_size,
                                      overlap=chunk_overlap)
                                  )

if not q_client.collection_exists(collection_name=os.environ.get('collection_name')):
    knowledge_base.load(recreate=False) # chroma here again 

# initialize agent
agent = Agent(knowledge=knowledge_base, search_knowledge=True, model=eval_llm)

# create the dataset for evaluation
eval_dataset = create_eval_ds(agent=agent, ground_truth_path=ground_truth_path) # make a dataset like in the screenshot and use that directly

# trigger evals
evaluation_dataset = EvaluationDataset.from_list(eval_dataset)
evaluator_llm = Azureopenai(llm=eval_llm) # replace the correct thing here for azure openai 
result = evaluate(dataset=evaluation_dataset, metrics=[Faithfulness(), ContextRelevance(),
                                                       ContextUtilization(), ContextRecall(),
                                                       FactualCorrectness()])

for score in result.scores:
    print(score)

# destroy the collection (for chroma)
if q_client.collection_exists(collection_name=os.environ.get('collection_name')):
    q_client.del