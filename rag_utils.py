import chromadb
import streamlit as st
from groq import Groq


@st.cache_resource
def get_embedding_model():
    """
    Load the embedding model only when a PDF is processed.
    """

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def create_vector_store(chunks):
    """
    Create a ChromaDB collection and store PDF chunks
    with their embeddings.
    """

    embedding_model = get_embedding_model()

    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="pdf_documents"
    )

    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

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

    embedding_model = get_embedding_model()

    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )

    return results["documents"][0]


def generate_answer(question, retrieved_chunks):
    """
    Generate an answer using only retrieved PDF context.
    """

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured in Streamlit secrets."
        )

    client = Groq(api_key=api_key)

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, respond exactly:

I don't know based on the provided context

Do not use outside knowledge.

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