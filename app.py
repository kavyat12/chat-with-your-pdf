import streamlit as st

from pdf_utils import extract_text_from_pdf, chunk_text
from rag_utils import create_vector_store, ask_question


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Chat With Your PDF",
    page_icon="PDF",
    layout="wide"
)


# --------------------------------------------------
# Application title
# --------------------------------------------------

st.title("Chat With Your PDF")

st.write(
    "Upload a PDF and ask questions about its content. "
    "The answers are generated using only the information "
    "retrieved from your uploaded PDF."
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection" not in st.session_state:
    st.session_state.collection = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("Instructions")

    st.write(
        "1. Upload a PDF.\n\n"
        "2. Wait for the PDF to be processed.\n\n"
        "3. Ask questions about the PDF.\n\n"
        "4. The system retrieves relevant PDF content.\n\n"
        "5. The LLM generates an answer using the retrieved context."
    )

    st.divider()

    st.subheader("Model")

    st.write("Embedding model:")
    st.code("all-MiniLM-L6-v2")

    st.write("LLM:")
    st.code("openai/gpt-oss-20b")

    st.divider()

    # New Chat clears only conversation history
    if st.button("New Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# PDF upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# --------------------------------------------------
# Process uploaded PDF
# --------------------------------------------------

if uploaded_file is not None:

    # Process only when a new PDF is uploaded
    if st.session_state.pdf_name != uploaded_file.name:

        # New PDF means a completely new conversation
        st.session_state.messages = []
        st.session_state.collection = None
        st.session_state.pdf_name = uploaded_file.name

        with st.spinner("Reading and processing your PDF..."):

            try:

                # Extract text from PDF
                text = extract_text_from_pdf(uploaded_file)

                if not text.strip():

                    st.error(
                        "No extractable text was found in this PDF. "
                        "The PDF may be scanned or image-based."
                    )

                    st.session_state.collection = None
                    st.stop()

                # Split extracted text into overlapping chunks
                chunks = chunk_text(
                    text,
                    chunk_size=250,
                    overlap=50
                )

                if not chunks:

                    st.error(
                        "The PDF could not be divided into text chunks."
                    )

                    st.session_state.collection = None
                    st.stop()

                # Create ChromaDB vector store
                collection = create_vector_store(chunks)

                # Store vector collection
                st.session_state.collection = collection

                st.success(
                    f"PDF processed successfully. "
                    f"Created {len(chunks)} text chunks."
                )

            except Exception as e:

                st.session_state.collection = None

                st.error(
                    f"An error occurred while processing the PDF: {e}"
                )

                st.stop()

    else:

        # Current PDF is already processed
        if st.session_state.collection is not None:

            st.success(
                f"PDF ready: {st.session_state.pdf_name}"
            )


# --------------------------------------------------
# Chat history
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# Chat input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your PDF..."
)


# --------------------------------------------------
# Question processing
# --------------------------------------------------

if question:

    # Make sure a processed PDF is available
    if st.session_state.collection is None:

        st.warning(
            "Please upload and process a PDF before asking a question."
        )

        st.stop()

    # Add user message to history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the PDF and generating an answer..."
        ):

            try:

                answer, retrieved_chunks = ask_question(
                    st.session_state.collection,
                    question,
                    top_k=3
                )

                st.markdown(answer)

                # Show retrieved PDF content
                with st.expander(
                    "View retrieved PDF content"
                ):

                    for i, chunk in enumerate(
                        retrieved_chunks,
                        start=1
                    ):

                        st.markdown(
                            f"**Retrieved chunk {i}**"
                        )

                        st.write(chunk)

                        if i < len(retrieved_chunks):

                            st.divider()

                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                error_message = (
                    f"An error occurred while generating the answer: {e}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )