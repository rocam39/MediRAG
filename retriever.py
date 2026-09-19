import os

from loader import load_pdf
from chunker import chunk_pages
from embeddings import EmbeddingModel
from vector_store import VectorStore


class Retriever:

    def __init__(
        self,
        documents_path="data/documents",
        knowledge_base_path="knowledge_base"
    ):

        self.documents_path = documents_path
        self.knowledge_base_path = knowledge_base_path

        # Load embedding model
        print("\nLoading embedding model...")

        self.embedding_model = EmbeddingModel()

        # Initialize vector store
        self.vector_store = VectorStore()

        # Try loading existing knowledge base
        loaded = self.vector_store.load(
            self.knowledge_base_path
        )

        if loaded:

            print("\nExisting knowledge base loaded.")

            print(
                f"Total chunks: "
                f"{len(self.vector_store.chunks)}"
            )

        else:

            print(
                "\nNo existing knowledge base found."
            )

            self.build()

    # ======================================================
    # BUILD KNOWLEDGE BASE
    # ======================================================

    def build(self):

        print("\n" + "=" * 60)
        print("BUILDING KNOWLEDGE BASE")
        print("=" * 60)

        if not os.path.exists(
            self.documents_path
        ):

            raise ValueError(
                f"Documents directory not found: "
                f"{self.documents_path}"
            )

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

        # --------------------------------------------------
        # Load PDFs
        # --------------------------------------------------

        for pdf_path in pdf_files:

            filename = os.path.basename(
                pdf_path
            )

            print(
                f"\nLoading: {filename}"
            )

            pages = load_pdf(
                pdf_path
            )

            chunks = chunk_pages(
                pages
            )

            # Add source metadata
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

        # --------------------------------------------------
        # Generate embeddings
        # --------------------------------------------------

        texts = [
            chunk["text"]
            for chunk in all_chunks
        ]

        print(
            "\nGenerating embeddings..."
        )

        embeddings = (
            self.embedding_model.encode(
                texts
            )
        )

        # --------------------------------------------------
        # Create FAISS index
        # --------------------------------------------------

        print(
            "\nCreating FAISS index..."
        )

        self.vector_store = VectorStore(
            embeddings.shape[1]
        )

        self.vector_store.add(
            embeddings,
            all_chunks
        )

        # --------------------------------------------------
        # Save knowledge base
        # --------------------------------------------------

        self.vector_store.save(
            self.knowledge_base_path
        )

        print(
            "\nKnowledge base built successfully."
        )

        print(
            f"Stored {len(all_chunks)} chunks."
        )

    # ======================================================
    # RETRIEVE
    # ======================================================

    def retrieve(
        self,
        query,
        top_k=5,
        threshold=0.0
    ):

        print("\n" + "=" * 60)
        print("RETRIEVING DOCUMENTS")
        print("=" * 60)

        print(
            f"Query: {query}"
        )

        # --------------------------------------------------
        # Generate query embedding
        # --------------------------------------------------

        query_embedding = (
            self.embedding_model.encode(
                [query]
            )
        )

        # --------------------------------------------------
        # FAISS similarity search
        # --------------------------------------------------

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            threshold=threshold
        )

        # --------------------------------------------------
        # Debug output
        # --------------------------------------------------

        print(
            f"\nRetrieved {len(results)} chunks."
        )

        for i, result in enumerate(
            results,
            start=1
        ):

            chunk = result["chunk"]

            print(
                f"\n--- Result {i} ---"
            )

            print(
                f"Similarity: "
                f"{result['score']:.4f}"
            )

            print(
                f"Source: "
                f"{chunk.get('source', 'Unknown')}"
            )

            print(
                f"Page: "
                f"{chunk.get('page', 'Unknown')}"
            )

            print(
                f"{chunk['text'][:500]}"
            )

        print(
            "=" * 60
        )

        return results
