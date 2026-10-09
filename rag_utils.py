```python
import os

import chromadb
import streamlit as st
from groq import Groq


@st.cache_resource
def get_embedding_model():
    """
    Load the embedding model only when needed.
    """

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def create_vector_store(chunks):
    """
    Create a ChromaDB collection and store PDF chunks
    with their embeddings.
    """

    if not chunks:
        raise ValueError("No text chunks are available to store.")

    embedding_model = get_embedding_model()

    # Create a fresh in-memory ChromaDB client for this PDF.
    client = chromadb.Client()

    collection = client.create_collection(
        name="pdf_" + os.urandom(8).hex()
    )

    embeddings = embedding_model.encode(chunks).tolist()

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    return collection


def retrieve_chunks(collection, question, top_k=3):
    """
    Retrieve the most relevant PDF chunks.
    """

    if not question.strip():
        return []

    embedding_model = get_embedding_model()

    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    # Never request more chunks than the collection contains.
    count = collection.count()

    if count == 0:
        return []

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(top_k, count)
    )

    return results.get("documents", [[]])[0]


def get_groq_api_key():
    """
    Read the API key from the environment first,
    then from Streamlit Secrets.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    return api_key


def generate_answer(question, retrieved_chunks):
    """
    Generate an answer grounded in retrieved PDF context.
    """

    api_key = get_groq_api_key()

    if not api_key:
        raise ValueError(
            "Groq API key is missing. Configure GROQ_API_KEY "
            "in Streamlit Cloud Secrets or the hosting environment."
        )

    if not retrieved_chunks:
        return "I don't know based on the provided context"

    client = Groq(api_key=api_key)

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, respond exactly:

I don't know based on the provided context

Do not use outside knowledge. If the context does not
support an answer, use the exact refusal above.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


def ask_question(collection, question, top_k=3):
    """
    Retrieve relevant chunks and generate a grounded answer.
    """

    retrieved_chunks = retrieve_chunks(
        collection,
        question,
        top_k
    )

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    return answer, retrieved_chunks
```
