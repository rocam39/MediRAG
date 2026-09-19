from retriever import Retriever

retriever = Retriever(
    "data/Diabetes.pdf"
)

query = "How to get rid of addictions?"

results = retriever.retrieve(
    query,
    top_k=5,
    threshold=0.5)

print("\nQUERY:")
print(query)

print("\nRETRIEVED CHUNKS:")


if not results:
    print("\nNo sufficiently relevant information found in the documents.")

else:
    for i, result in enumerate(results):

        print(f"\n--- Result {i + 1} ---")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Page: {result['chunk']['page']}")
        print(result["chunk"]["text"])