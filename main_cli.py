from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
import os
import argparse

# Function to load PDFs from a directory
def load_documents(directory):
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
                # Add more file types as needed
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
    return documents

# Load PDFs
documents = load_documents('./documents')

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
texts = text_splitter.split_documents(documents)

# Create embeddings using Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create and persist vector store
db = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"  # This will persist your database locally
)
db.persist()

# Create LLM instance
llm = Ollama(model="llama3")

# Create retrieval chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 most relevant chunks
)

# Function to query the documents
def query_documents(question):
    response = qa.run(question)
    return response

def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Query documents using RAG')
    parser.add_argument('--setup', action='store_true', 
                       help='Setup/rebuild the document database')
    parser.add_argument('--query', type=str, 
                       help='Query to ask about the documents')
    
    args = parser.parse_args()

    if args.setup:
        # Load and process documents
        documents = load_documents('./documents')
        texts = text_splitter.split_documents(documents)
        
        # Create and persist vector store
        db = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        db.persist()
        print("Database setup complete!")

    elif args.query:
        # Load existing database
        db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        
        # Setup QA chain
        llm = Ollama(model="llama3")
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3})
        )
        
        # Get answer
        answer = qa.run(args.query)
        print("\nQuestion:", args.query)
        print("\nAnswer:", answer)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()