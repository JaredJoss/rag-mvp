# Enhanced Document Chat

## Overview

Enhanced Document Chat is a Streamlit application that allows users to upload documents and interact with them through a question-answering interface. Utilizing LangChain and Ollama, the app processes PDF and TXT files, enabling users to ask questions and receive detailed answers along with the sources of information.

## Features
- Upload documents in PDF and TXT formats.
- Enhanced document preprocessing for improved text clarity.
- Better text chunking with adjustable parameters.
- MMR (Maximum Marginal Relevance)-based retrieval for relevant responses.
- Source tracking to show where information is derived from.
- Improved error handling for a smoother user experience.
- Response caching to optimize performance.
- Temperature-controlled responses for varied answer styles.

## Requirements
To run this application, ensure you have the following packages installed:

- Streamlit
- LangChain
- Ollama
- langchain_community
- You can install the required packages using pip:

```bash
pip install streamlit langchain ollama langchain_community
```

## How to Run
1. Clone the repository or download the script.
2. Navigate to the directory containing the script.
3. Run the Streamlit application with the following command:
```bash
streamlit run app_v2.py
```

4. Open your web browser and go to http://localhost:8501.

## Usage
Upload Documents: Click on the "Upload your documents" button to select PDF or TXT

## To do:
- [ ] How does it scale to multiple documents.
- [ ] Sometime it does not get the answer, can I ask it to try 3 times before it says "I don't know".
- [ ] Try other embedding models.
- [ ] Try other LLMs.
- [ ] Try a different prompt.
- [ ] Use docker to deploy so can use Ollama in the .
- [ ] Deploy to cloud.