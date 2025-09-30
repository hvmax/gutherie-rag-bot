# --- RAG Chatbot with Groq API and Streamlit UI ---

# This script creates a web interface for the RAG chatbot using Streamlit.
# It includes a chat history that persists across user interactions.

# --- 1. Import Necessary Libraries ---
import os
import torch
import traceback
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- App Configuration ---
st.set_page_config(page_title="Document Chatbot", page_icon="🤖")
st.title("🤖 Chat with Your Documents")
st.write("This chatbot uses your local PDF files to answer questions. It's powered by Groq for super-fast responses.")
st.write("Place your PDF files in a folder named 'data' in the same directory as this script.")

# --- Configuration & Caching ---
# Use Streamlit's caching to load the model and index only once.
@st.cache_resource
def load_components():
    """Loads all the necessary components for the RAG pipeline."""
    # --- Configuration ---
    EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
    # Dynamically create the index path based on the model name
    FAISS_INDEX_PATH = f"faiss_index_{EMBEDDING_MODEL_NAME.replace('/', '_')}"
    DATA_PATH = "./data/"

    # --- Initialize the embeddings model ---
    # Using 'mps' for Mac, 'cuda' for NVIDIA, or 'cpu'
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )

    # --- Load the vector store if it exists, otherwise create it ---
    if os.path.exists(FAISS_INDEX_PATH):
        st.info(f"Loading existing vector store from: {FAISS_INDEX_PATH}")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        st.success("Vector store loaded successfully!")
    else:
        # One-time setup: build and save the index
        with st.spinner("No existing index found. Building a new one from documents in './data/'... This may take a few minutes."):
            if not os.path.exists(DATA_PATH):
                st.error(f"Error: The directory '{DATA_PATH}' was not found. Please create it and add your PDF files.")
                st.stop()
            
            # Use unstructured for robust PDF processing
            loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", show_progress=True)
            documents = loader.load()
            
            if not documents:
                st.error("No documents were found in the './data/' directory. Please add your PDFs and restart.")
                st.stop()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            
            vector_store = FAISS.from_documents(texts, embeddings)
            vector_store.save_local(FAISS_INDEX_PATH)
            st.success("New index built and saved successfully!")
    
    return vector_store

@st.cache_resource
def get_qa_chain(_vector_store):
    """Initializes and returns the RAG chain."""
    # Connect to the Groq API (uses GROQ_API_KEY from st.secrets)
    llm = ChatGroq(
        temperature=0.1,
        model_name="llama3-70b-8192", # Updated to a recommended Groq model
        api_key=st.secrets["GROQ_API_KEY"]
    )

    prompt_template = """
    Use the following pieces of context to answer the user's question.
    Provide a concise and factual summary based ONLY on the text provided.
    If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.

    After your summary, you MUST include a "Confidence Score". This score should be a percentage (e.g., 95%) that estimates how well the provided context answered the user's question.
    The confidence score should be based on the relevance and completeness of the context in relation to the question.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vector_store.as_retriever(search_kwargs={'k': 6}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True # Optional: to see which chunks were used
    )
    return qa_chain

# --- Main App Logic ---
try:
    vector_store = load_components()
    qa_chain = get_qa_chain(vector_store)
except Exception as e:
    st.error("There was an error initializing the chatbot. Please check the console for details.")
    st.error(e)
    # Print detailed error to console for debugging
    traceback.print_exc()
    st.stop()

# --- Chat History Management ---
# Use Streamlit's session_state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Chat Logic ---
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.invoke(prompt)
                st.markdown(response['result'])
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response['result']})
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})