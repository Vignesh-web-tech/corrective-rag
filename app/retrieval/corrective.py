def evaluate_retrieval(results, threshold=0.40):

    if not results:
        return {
            "status": "INCORRECT",
            "score": 0.0
        }

    best_score = max(
        item["score"]
        for item in results
    )

    if best_score >= threshold:
        return {
            "status": "CORRECT",
            "score": best_score
        }

    return {
        "status": "INCORRECT",
        "score": best_score
    }
    
def corrective_retrieval(
    query,
    serialized_file,
    top_k=5
):
    """
    First retrieval failed.
    Perform a broader retrieval.
    """

    from app.retrieval.retriever import retrieve

    results = retrieve(
        query=query,
        serialized_file=serialized_file,
        top_k=top_k
    )

    return results    