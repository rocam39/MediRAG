import json
import os

import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension=None):

        self.index = None
        self.chunks = []

        if dimension is not None:

            self.index = faiss.IndexFlatIP(
                dimension
            )


    def add(self, embeddings, chunks):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        if self.index is None:

            dimension = embeddings.shape[1]

            self.index = faiss.IndexFlatIP(
                dimension
            )

        self.index.add(embeddings)

        self.chunks.extend(chunks)


    def search(
        self,
        query_embedding,
        top_k=5,
        threshold=0.4
    ):

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        if self.index is None:
            return []

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if (
                index != -1
                and score >= threshold
            ):

                results.append({
                    "score": float(score),
                    "chunk": self.chunks[index]
                })

        return results


    def save(self, directory):

        os.makedirs(
            directory,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            os.path.join(
                directory,
                "index.faiss"
            )
        )

        with open(
            os.path.join(
                directory,
                "chunks.json"
            ),
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.chunks,
                file,
                ensure_ascii=False,
                indent=2
            )


    def load(self, directory):

        index_path = os.path.join(
            directory,
            "index.faiss"
        )

        chunks_path = os.path.join(
            directory,
            "chunks.json"
        )

        if not os.path.exists(index_path):
            return False

        if not os.path.exists(chunks_path):
            return False

        self.index = faiss.read_index(
            index_path
        )

        with open(
            chunks_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.chunks = json.load(file)

        return True