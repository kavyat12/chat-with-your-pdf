# Chat With Your PDF

Chat With Your PDF is a Streamlit-based RAG application that allows users to upload a PDF and ask questions about its content.

The application extracts text from the uploaded PDF, divides the text into overlapping chunks, creates embeddings using a sentence-transformer model, stores the embeddings in ChromaDB, retrieves the most relevant content for a question, and generates a grounded answer using the Groq API.

The application is designed to answer questions using only the information available in the uploaded PDF.

---

## Features

- Upload PDF documents
- Extract text from PDF files
- Split PDF text into overlapping chunks
- Generate text embeddings
- Store document embeddings in ChromaDB
- Retrieve relevant PDF content using semantic search
- Generate answers using the Groq API
- Chat-style interface using Streamlit
- Maintain conversation history during the session
- Display retrieved PDF content used for answering
- Refuse questions when the answer is not available in the PDF
- Detect PDFs with no extractable text
- Start a new chat without removing the currently loaded PDF
- Upload a different PDF and automatically process it

---

## Technologies Used

- Python
- Streamlit
- Groq API
- Llama / Groq-hosted LLM
- sentence-transformers
- ChromaDB
- pypdf
- python-dotenv

### Embedding Model

```text
all-MiniLM-L6-v2