import os

from chunker import chunk_pages
from embeddings import EmbeddingModel
from loader import load_pdf
from vector_store import VectorStore


class Retriever:

    def __init__(
        self,
        documents_path="data/documents",
        knowledge_base_path="knowledge_base"
    ):

        self.documents_path = documents_path
        self.knowledge_base_path = knowledge_base_path

        self.embedding_model = EmbeddingModel()

        self.vector_store = VectorStore()

        # Try loading existing knowledge base
        loaded = self.vector_store.load(
            self.knowledge_base_path
        )

        if loaded:

            print(
                "Existing knowledge base loaded."
            )

            print(
                f"Total chunks: "
                f"{len(self.vector_store.chunks)}"
            )

        else:

            print(
                "No existing knowledge base found."
            )

            self.build()


    def build(self):

        print("\nBuilding knowledge base...")

        pdf_files = [
            os.path.join(
                self.documents_path,
                file
            )
            for file in os.listdir(
                self.documents_path
            )
            if file.lower().endswith(".pdf")
        ]

        if not pdf_files:

            raise ValueError(
                "No PDF files found in "
                f"{self.documents_path}"
            )

        print(
            f"Found {len(pdf_files)} PDF files."
        )

        all_chunks = []

        for pdf_path in pdf_files:

            filename = os.path.basename(
                pdf_path
            )

            print(
                f"Loading: {filename}"
            )

            pages = load_pdf(
                pdf_path
            )

            chunks = chunk_pages(
                pages
            )

            # Store source filename
            for chunk in chunks:

                chunk["source"] = filename

            all_chunks.extend(
                chunks
            )

            print(
                f"  → {len(chunks)} chunks"
            )

        print(
            f"\nTotal chunks: "
            f"{len(all_chunks)}"
        )

        texts = [
            chunk["text"]
            for chunk in all_chunks
        ]

        print(
            "\nGenerating embeddings..."
        )

        embeddings = self.embedding_model.encode(
            texts
        )

        print(
            "Creating FAISS index..."
        )

        self.vector_store = VectorStore(
            embeddings.shape[1]
        )

        self.vector_store.add(
            embeddings,
            all_chunks
        )

        self.vector_store.save(
            self.knowledge_base_path
        )

        print(
            "\nKnowledge base built successfully."
        )


    def retrieve(
        self,
        query,
        top_k=5,
        threshold=0.5
    ):

        query_embedding = (
            self.embedding_model.encode(
                [query]
            )
        )

        results = self.vector_store.search(
            query_embedding,
            top_k,
            threshold
        )

        return results