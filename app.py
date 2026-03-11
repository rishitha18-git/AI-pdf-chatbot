# =========================
# IMPORTS
# =========================
import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# =========================
# STEP 0: LOAD LOCAL LLM (ONCE)
# =========================
model_id = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
)

llm = HuggingFacePipeline(pipeline=pipe)


# =========================
# STREAMLIT UI
# =========================
st.title("AI PDF Chatbot")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")


# =========================
# PDF PROCESSING
# =========================
if pdf_file:
    # STEP 1: EXTRACT TEXT
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    st.success("PDF text extracted")

    # STEP 2: SPLIT TEXT
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    st.success(f"Text split into {len(chunks)} chunks")

    # STEP 3: CREATE EMBEDDINGS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.write("Embedding size:", len(embeddings.embed_query("hello")))

    # STEP 4: CREATE FAISS VECTOR STORE
    vector_store = FAISS.from_texts(chunks, embeddings)
    retriever = vector_store.as_retriever()

    st.success("FAISS vector store created successfully!")

    # =========================
    # QUESTION ANSWERING
    # =========================
    st.subheader("Ask questions about your PDF")

    user_question = st.text_input("Your question:")

    if user_question:
        # Retrieve relevant chunks
        docs = retriever.invoke(user_question)

        # Build context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Prompt
        prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{user_question}

Answer:
"""

        # Run LLM
        answer = llm.invoke(prompt)

        st.write("### Answer")
        st.write(answer)

        with st.expander("Source Chunks"):
            for doc in docs:
                st.write(doc.page_content)
