import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma, FAISS
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.prompts import PromptTemplate
import os
import re

# Set page configuration
st.set_page_config(
    page_title="Enhanced Document Chat",
    page_icon="📚",
    layout="wide"
)

# Add title and description
st.title("💬 Enhanced Document Chat")
st.write("Upload your documents and ask questions about them!")

def preprocess_text(text: str) -> str:
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove special characters
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text

def validate_question(question: str) -> bool:
    if len(question.strip()) < 3:
        st.error("Please enter a longer question")
        return False
    return True

@st.cache_resource
def load_and_process_documents(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif file.endswith('.txt'):
                    loader = TextLoader(file_path)
                else:
                    continue
                    
                docs = loader.load()
                # Preprocess each document
                for doc in docs:
                    doc.page_content = preprocess_text(doc.page_content)
                documents.extend(docs)
                
            except Exception as e:
                st.error(f"Error loading {file_path}: {str(e)}")
    
    # Improved text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len,
        keep_separator=True
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # # Create vector store
    # db = Chroma.from_documents(
    #     documents=texts,
    #     embedding=embeddings,
    #     persist_directory="./chroma_db"
    # )
    # db.persist()

    # return db

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Save the index locally
    vectorstore.save_local("faiss_index")
    
    return vectorstore

@st.cache_data
def get_qa_response(_qa, question: str):
    return qa({"question": question})

# Enhanced prompt template
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

Context: {summaries}

Question: {question}

Give a detailed answer and explain your reasoning step by step. 
If you use information from the context, indicate which part you're referring to.

Answer: """

PROMPT = PromptTemplate(
    template=template, 
    input_variables=["summaries ", "question"]
)

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
    
    # # Load and process documents
    # db = load_and_process_documents('./documents')
    
    # # Create improved retriever
    # retriever = db.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={
    #         "k": 5,
    #         "fetch_k": 8,
    #         "lambda_mult": 0.7
    #     }
    # )

    vectorstore = load_and_process_documents('./documents')
    
    # Create LLM instance with temperature control
    llm = Ollama(
        model="llama3-chatqa",
        temperature=0.3
    )
    
    # # Create enhanced QA chain
    # qa = RetrievalQAWithSourcesChain.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=retriever,
    #     return_source_documents=True,
    #     chain_type_kwargs={
    #     "prompt": PROMPT,
    #     "document_variable_name": "summaries"
    #     }
    # )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 8,
                "lambda_mult": 0.7
            }
        ),
        chain_type_kwargs={
            "prompt": PROMPT,
            "document_variable_name": "summaries"
            },
        return_source_documents=True
    )
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input with improved error handling
    if prompt := st.chat_input("Ask a question about your documents"):
        if validate_question(prompt):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # result = get_qa_response(qa, prompt)
                        result = qa(prompt)
                        
                        # Display answer
                        st.markdown(result["result"])
                        
                        # Display sources
                        st.markdown("**Sources:**")
                        for source in result["source_documents"]:
                            st.markdown(f"- {source.metadata.get('source', 'Unknown source')}")
                        
                        # Save to chat history
                        full_response = f"{result['result']}\n\n**Sources:**\n" + \
                            "\n".join([f"- {source.metadata.get('source', 'Unknown source')}" 
                                     for source in result["source_documents"]])
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

else:
    st.info("Please upload your documents to start chatting!")

# Add sidebar with additional information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This enhanced RAG application features:
    
    - Improved document preprocessing
    - Better text chunking
    - MMR-based retrieval
    - Source tracking
    - Error handling
    - Response caching
    - Temperature-controlled responses
    
    Upload PDF and TXT files to begin!
    """)