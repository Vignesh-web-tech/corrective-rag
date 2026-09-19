import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

SERIALIZED_DIR = BASE_DIR / "data" / "serialized"

SERIALIZED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_document(
    document_id,
    chunks,
    embeddings,
    metadata
):

    data = {
        "document_id": document_id,
        "chunks": chunks,
        "embeddings": embeddings,
        "metadata": metadata
    }

    file_path = (
        SERIALIZED_DIR /
        f"{document_id}.pkl"
    )

    with open(
        file_path,
        "wb"
    ) as file:

        pickle.dump(
            data,
            file
        )

    return str(file_path)