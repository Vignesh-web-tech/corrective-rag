import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def retrieve(
    query: str,
    serialized_file: str,
    top_k: int = 5
):

    with open(
        serialized_file,
        "rb"
    ) as file:

        data = pickle.load(file)

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    embeddings = np.array(
        data["embeddings"]
    )

    scores = embeddings @ query_embedding

    indexes = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in indexes:

        results.append({
            "chunk": data["chunks"][index],
            "score": float(scores[index])
        })

    return results