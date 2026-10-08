# Final Reflection

## What I Built

For the Week 4 project, I built a Streamlit application called **Chat With Your PDF**.

The application allows a user to upload a PDF document and ask questions about its content through a conversational interface. The application uses a Retrieval-Augmented Generation (RAG) pipeline to retrieve relevant information from the uploaded PDF and provide grounded answers.

## RAG Pipeline

The application follows these main steps:

1. The user uploads a PDF document.
2. Text is extracted from the PDF using `pypdf`.
3. The extracted text is divided into overlapping chunks.
4. Each chunk is converted into an embedding using the `all-MiniLM-L6-v2` sentence-transformer model.
5. The embeddings and document chunks are stored in ChromaDB.
6. When the user asks a question, the question is converted into an embedding.
7. ChromaDB retrieves the most relevant document chunks.
8. The retrieved chunks are provided as context to the Groq-hosted LLM.
9. The LLM generates an answer using the retrieved PDF content.

## Chunking and Overlap

I used approximately 250 words per chunk with an overlap of 50 words.

The overlap helps preserve information when an important sentence or concept is located near the boundary between two chunks. Without overlap, some context could be separated between chunks and become more difficult to retrieve correctly.

## What I Learned About Grounding

One of the main things I learned from this project is the importance of grounding an LLM response in retrieved information.

Instead of allowing the model to answer using general knowledge, the application provides relevant content from the uploaded PDF as context. The prompt instructs the model to answer only from the provided context.

If the answer is not available in the retrieved PDF content, the application instructs the model to respond:

`I don't know based on the provided context`

This helps reduce answers that are not supported by the uploaded document.

## ChromaDB vs Manual Similarity Search

In the previous project, similarity search was implemented manually using embeddings and cosine similarity.

For this project, I used ChromaDB as the vector database. ChromaDB provides functionality for storing embeddings and documents and retrieving relevant results through vector search.

Using ChromaDB makes the retrieval part of the RAG pipeline easier to manage and keeps the vector search logic separate from the rest of the application.

## Streamlit Application

I also learned how to build a complete user interface using Streamlit.

The application includes:

- PDF upload
- PDF processing feedback
- Chat interface
- Conversation history
- Question input
- Loading indicators
- Error messages
- Retrieved PDF content
- New Chat functionality
- Sidebar instructions and model information

The conversation history is maintained using Streamlit session state.

## Error Handling

The application handles several common situations.

If a user asks a question before uploading a PDF, the application asks the user to upload a PDF first.

If a PDF does not contain extractable text, the application informs the user that the PDF may be scanned or image-based.

The application also checks whether the Groq API key is configured and displays an error if there is a problem while processing the PDF or generating an answer.

## Testing

I tested the application by uploading PDF documents and asking questions related to their content.

I also tested questions that were unrelated to the uploaded document. The application correctly follows the grounding instruction and returns:

`I don't know based on the provided context`

when the required information is not available in the provided context.

## Final Learning

This project helped me understand how a complete RAG application works from document ingestion to final answer generation.

I learned how PDF extraction, text chunking, embeddings, vector databases, semantic retrieval, prompting, LLM generation, and a web interface can be combined into one working application.

The project also helped me understand that retrieval quality and the quality of the provided context are important for producing useful and grounded answers.