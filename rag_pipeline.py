from generator import LLMGenerator
from retriever import Retriever


class RAGPipeline:

    def __init__(
        self,
        documents_path="data/documents",
        knowledge_base_path="knowledge_base"
    ):

        print(
            "========================================"
        )

        print(
            "Initializing MediRAG"
        )

        print(
            "========================================"
        )

        self.retriever = Retriever(
            documents_path,
            knowledge_base_path
        )

        self.llm = LLMGenerator()


    def answer(
        self,
        query,
        top_k=5,
        threshold=0.5
    ):

        results = self.retriever.retrieve(
            query,
            top_k=top_k,
            threshold=threshold
        )

        # No relevant information
        if not results:

            return (
                "I couldn't find this information in the provided documents.",
                []
            )


        context_parts = []

        for i, result in enumerate(results):

            chunk = result["chunk"]

            source_number = i + 1

            context_parts.append(
                f"[Source {source_number}]\n"
                f"Document: {chunk['source']}\n"
                f"Page: {chunk['page']}\n"
                f"{chunk['text']}"
            )

        context = "\n\n".join(
            context_parts
        )

        answer = self.llm.generate(
            query,
            context
        )

        return answer, results