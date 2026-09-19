from embeddings import EmbeddingModel

model = EmbeddingModel()

texts = [
    "Diabetes causes high blood sugar levels.",
    "Diabetes can lead to elevated glucose in the blood.",
    "The car has a powerful diesel engine."
]

vectors = model.encode(texts)

print("Number of vectors:", len(vectors))
print("Vector dimensions:", vectors.shape)

print("\nFirst vector:")
print(vectors[0])
