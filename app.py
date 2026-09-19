import os
import streamlit as st

from rag_pipeline import RAGPipeline


DOCUMENTS_PATH = "data/documents"
KNOWLEDGE_BASE_PATH = "knowledge_base"


st.set_page_config(
    page_title="MediRAG",
    page_icon="🩺",
    layout="wide"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🩺 MediRAG")

st.caption(
    "Medical knowledge assistant powered by "
    "Retrieval-Augmented Generation"
)

st.info(
    "MediRAG is an educational information tool. "
    "It retrieves information from the provided documents "
    "and is not a substitute for professional medical advice."
)


# --------------------------------------------------
# Directories
# --------------------------------------------------

os.makedirs(DOCUMENTS_PATH, exist_ok=True)
os.makedirs(KNOWLEDGE_BASE_PATH, exist_ok=True)


# --------------------------------------------------
# Initialize RAG pipeline
# --------------------------------------------------

if "pipeline" not in st.session_state:

    with st.spinner("Loading MediRAG..."):

        st.session_state.pipeline = RAGPipeline(
            DOCUMENTS_PATH,
            KNOWLEDGE_BASE_PATH
        )


# --------------------------------------------------
# Conversation
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📚 Knowledge Base")

    pdf_files = [
        file
        for file in os.listdir(DOCUMENTS_PATH)
        if file.lower().endswith(".pdf")
    ]

    st.write(
        f"**{len(pdf_files)} PDF(s)** available"
    )

    if pdf_files:

        with st.expander("View documents"):

            for file in sorted(pdf_files):
                st.write(f"• {file}")

    st.divider()

    st.header("⚙️ Retrieval Settings")

    top_k = st.slider(
        "Number of sources",
        min_value=1,
        max_value=10,
        value=5
    )

    threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    st.divider()

    if st.button("🗑️ Clear Conversation"):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📚 Sources"):

                for i, source in enumerate(
                    message["sources"],
                    start=1
                ):

                    st.markdown(
                        f"**[{i}] "
                        f"{source['source']} — "
                        f"Page {source['page']}**"
                    )

                    st.caption(
                        f"Similarity: "
                        f"{source['score']:.4f}"
                    )


# --------------------------------------------------
# Chat input
# --------------------------------------------------

query = st.chat_input(
    "Ask a question about the medical documents..."
)


if query:

    # User message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)


    # Assistant response

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                answer, results = (
                    st.session_state.pipeline.answer(
                        query,
                        top_k=top_k,
                        threshold=threshold
                    )
                )

            except Exception as error:

                st.error(
                    "An error occurred while processing "
                    "your question."
                )

                st.exception(error)

                answer = (
                    "I couldn't process your question "
                    "right now."
                )

                results = []


        st.markdown(answer)


        # Sources

        if results:

            with st.expander(
                f"📚 Sources ({len(results)})",
                expanded=True
            ):

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    chunk = result["chunk"]

                    st.markdown(
                        f"**[{i}] "
                        f"{chunk['source']} — "
                        f"Page {chunk['page']}**"
                    )

                    st.caption(
                        f"Similarity: "
                        f"{result['score']:.4f}"
                    )

                    st.markdown("---")


    # Save assistant response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": [
                {
                    "source": result["chunk"]["source"],
                    "page": result["chunk"]["page"],
                    "score": result["score"]
                }
                for result in results
            ]
        }
    )