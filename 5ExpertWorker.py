import os
import glob
from dotenv import load_dotenv
import gradio as gr

# LangChain Imports
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Ollama LLM import from LangChain
from langchain.llms import Ollama

# For debugging or streaming outputs (optional)
from langchain.callbacks import StdOutCallbackHandler

# -----------------------------------------------------------------------------
# Environment & Basic Configuration
# -----------------------------------------------------------------------------
# If you still have environment variables you’d like to load, do so here
load_dotenv()

# Adjust the Ollama model name to whatever you have installed, for example:
# "llama2" or "llama2-7b-chat" or "llama2-13b-chat" etc.
OLLAMA_MODEL = "llama3.2"

# Name of the local persistent Chroma DB folder
db_name = "vector_db"

# -----------------------------------------------------------------------------
# Document Loading
# -----------------------------------------------------------------------------
folders = glob.glob("knowledge-base/*")

def add_metadata(doc, doc_type):
    doc.metadata["doc_type"] = doc_type
    return doc

text_loader_kwargs = {'encoding': 'utf-8'}

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])

# -----------------------------------------------------------------------------
# Chunking Text
# -----------------------------------------------------------------------------
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"Total number of chunks: {len(chunks)}")
print(f"Document types found: {set(doc.metadata['doc_type'] for doc in documents)}")

# -----------------------------------------------------------------------------
# Vector Store (Chroma) and Embeddings
# -----------------------------------------------------------------------------
# Using HuggingFace embeddings so we are fully local (no OpenAI needed).
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# If the Chroma DB folder already exists, you may want to recreate or delete it
if os.path.exists(db_name):
    # This is one way to remove the old DB. Another approach is to call
    # Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    import shutil
    shutil.rmtree(db_name)

# Create (and persist) vector store
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# Let's get some info on embeddings
collection = vectorstore._collection
count = collection.count()
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")

# -----------------------------------------------------------------------------
# Create Ollama LLM
# -----------------------------------------------------------------------------
# Refer to https://python.langchain.com/docs/modules/model_io/models/llms/integrations/ollama
# for additional parameters you can pass to Ollama (e.g. `temperature`, `context`, `f16_kv`, etc.)

llm = Ollama(
    model=OLLAMA_MODEL,
    base_url="http://localhost:11434",  # Adjust if your Ollama server is elsewhere
    temperature=0.7
)

# -----------------------------------------------------------------------------
# Set up Conversational Retrieval Chain
# -----------------------------------------------------------------------------
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
retriever = vectorstore.as_retriever()

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
)

# Test a simple query
query = "Please explain what Insurellm is in a couple of sentences"
result = conversation_chain.invoke({"question": query})
print("Answer:", result["answer"])

# -----------------------------------------------------------------------------
# Re-initialize memory for a fresh conversation if desired
# -----------------------------------------------------------------------------
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
)

def chat(question, history):
    # Simple chat function that uses the conversation chain
    result = conversation_chain.invoke({"question": question})
    return result["answer"]

# -----------------------------------------------------------------------------
# Gradio Chat Interface
# -----------------------------------------------------------------------------
view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)

# -----------------------------------------------------------------------------
# Demonstration of Callbacks to see what is going on under the hood (optional)
# -----------------------------------------------------------------------------
debug_llm = Ollama(
    model=OLLAMA_MODEL,
    base_url="http://localhost:11434",
    temperature=0.7
)

debug_memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
debug_retriever = vectorstore.as_retriever()

# Pass a StdOutCallbackHandler to see prompt/response in console
debug_chain = ConversationalRetrievalChain.from_llm(
    llm=debug_llm,
    retriever=debug_retriever,
    memory=debug_memory,
    callbacks=[StdOutCallbackHandler()]
)

query = "Who received the prestigious IIOTY award in 2023?"
result = debug_chain.invoke({"question": query})
print("\nAnswer:", result["answer"])

# -----------------------------------------------------------------------------
# Another example with k=25 retrieved documents
# -----------------------------------------------------------------------------
final_llm = Ollama(
    model=OLLAMA_MODEL,
    base_url="http://localhost:11434",
    temperature=0.7
)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=final_llm,
    retriever=retriever,
    memory=memory
)

def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    return result["answer"]

view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)
