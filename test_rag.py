from rag_pipeline import RAGPipeline

pipeline = RAGPipeline(
    "data/documents",
    "knowledge_base"
)


while True:

    print("\n" + "=" * 60)

    query = input(
        "Ask a question "
        "(type 'exit' to quit): "
    )

    if query.lower() == "exit":
        break

    answer, results = pipeline.answer(
        query,
        top_k=5,
        threshold=0.5
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(answer)

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    if not results:

        print(
            "No relevant sources found."
        )

    else:

        for i, result in enumerate(results):

            chunk = result["chunk"]

            print(
                f"\n[{i + 1}] "
                f"{chunk['source']} "
                f"(Page {chunk['page']})"
            )

            print(
                f"Similarity: "
                f"{result['score']:.4f}"
            )