from pypdf import PdfReader


def extract_text_from_pdf(pdf_file):
    """
    Extract text from all pages of the uploaded PDF.
    """

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text.strip()


def chunk_text(text, chunk_size=250, overlap=50):
    """
    Split text into overlapping word-based chunks.

    Example:
    chunk 1 -> words 1-250
    chunk 2 -> words 201-450
    chunk 3 -> words 401-650
    """

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks