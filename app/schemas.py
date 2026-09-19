from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    document_id: str