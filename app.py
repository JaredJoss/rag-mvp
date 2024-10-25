import streamlit as st
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
import os

# Set page configuration
st.set_page_config(
    page_title="Document Chat",
    page_icon="📚",
    layout="wide"
)

# Add title and description
st.title("💬 Chat with Your Documents")
st.write("Upload your documents and ask questions about them!")

# Function to load documents
@st.cache_resource
def load_and_process_documents(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file.endswith('.txt'):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
            except Exception as e:
                st.error(f"Error loading {file_path}: {str(e)}")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Smaller chunks
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    length_function=len
)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Create vector store
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    db.persist()
    
    return db

# File uploader
uploaded_files = st.file_uploader(
    "Upload your documents",
    type=['pdf', 'txt'],
    accept_multiple_files=True
)

if uploaded_files:
    # Create documents directory if it doesn't exist
    if not os.path.exists('./documents'):
        os.makedirs('./documents')
    
    # Save uploaded files
    for file in uploaded_files:
        with open(os.path.join('./documents', file.name), 'wb') as f:
            f.write(file.getbuffer())
    
    # Load and process documents
    db = load_and_process_documents('./documents')
    
    # Create LLM instance
    llm = Ollama(model="llama3-chatqa")
    
    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3})
    )
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Please upload your documents to start chatting!")

# Add sidebar with additional information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This application allows you to chat with your documents using RAG (Retrieval-Augmented Generation).
    
    **Features:**
    - Upload PDF and TXT files
    - Ask questions about your documents
    - Get AI-powered responses
    - Chat history
    """)