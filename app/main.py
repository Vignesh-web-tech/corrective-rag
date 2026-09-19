from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import uuid
from app.schemas import QueryRequest
from app.retrieval.retriever import retrieve
from app.retrieval.corrective import evaluate_retrieval
from app.ingestion.parser import parse_document
from app.ingestion.chunker import dynamic_chunk
from app.ingestion.embeddings import create_embeddings
from app.storage.vector_store import save_document
from app.generation.llm import generate_answer

app = FastAPI(
    title="Corrective RAG API"
)


UPLOAD_DIR = Path(
    "data/uploads"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


@app.get("/")
def home():

    return {
        "message": "Corrective RAG API is running"
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    document_id = str(
        uuid.uuid4()
    )

    file_path = (
        UPLOAD_DIR /
        f"{document_id}_{file.filename}"
    )

    content = await file.read()

    file_path.write_bytes(content)

    # 1. Parse
    text = parse_document(
        str(file_path)
    )

    # 2. Dynamic chunk
    chunks = dynamic_chunk(text)

    # 3. Embeddings
    embeddings = create_embeddings(
        chunks
    )

    # 4. Serialize
    serialized_file = save_document(
        document_id=document_id,
        chunks=chunks,
        embeddings=embeddings,
        metadata={
            "filename": file.filename,
            "content_type": file.content_type
        }
    )

    return {
        "status": "success",
        "document_id": document_id,
        "filename": file.filename,
        "chunks": len(chunks),
        "embedding_dimension": len(
            embeddings[0]
        ) if embeddings else 0,
        "serialized_file": serialized_file
    }

@app.post("/query")
async def query_rag(request: QueryRequest):

    query = request.query
    document_id = request.document_id

    serialized_file = (
        Path("data")
        / "serialized"
        / f"{document_id}.pkl"
    )

    # Document check
    if not serialized_file.exists():

        return {
            "status": "error",
            "message": "Document not found",
            "document_id": document_id
        }

    # --------------------------------
    # STEP 1: Initial Retrieval
    # --------------------------------

    results = retrieve(
        query=query,
        serialized_file=str(serialized_file),
        top_k=5
    )

    # --------------------------------
    # STEP 2: CRAG Evaluation
    # --------------------------------

    evaluation = evaluate_retrieval(
        results,
        threshold=0.40
    )

    # --------------------------------
    # STEP 3: Corrective Retrieval
    # --------------------------------

    if evaluation["status"] == "INCORRECT":

        print("Initial retrieval failed.")
        print("Running corrective retrieval...")

        results = corrective_retrieval(
            query=query,
            serialized_file=str(serialized_file),
            top_k=10
        )

        evaluation = evaluate_retrieval(
            results,
            threshold=0.30
        )

        retrieval_mode = "CORRECTIVE"

    else:

        retrieval_mode = "NORMAL"

    # --------------------------------
    # STEP 4: Convert chunks to context
    # --------------------------------

    context = "\n\n".join(
        result["chunk"]
        for result in results
    )

    # --------------------------------
    # STEP 5: Send context to Groq
    # --------------------------------

    answer = generate_answer(
        query=query,
        context=context
    )

    # --------------------------------
    # STEP 6: Final Response
    # --------------------------------

    return {
        "query": query,
        "retrieval_mode": retrieval_mode,
        "retrieval_status": evaluation,
        "answer": answer,
        "sources": results
    }