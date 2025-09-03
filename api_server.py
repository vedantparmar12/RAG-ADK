from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import json
import os
from datetime import datetime

from rag_agent import RAGAgent
from rag_agent.config import settings

app = FastAPI(title="RAG Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_agent = RAGAgent()

class CorpusCreate(BaseModel):
    name: str
    description: Optional[str] = None

class QueryRequest(BaseModel):
    corpus_id: str
    query: str
    num_results: Optional[int] = 5

class CorpusResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    document_count: int
    created_at: Optional[str]
    updated_at: Optional[str]

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float

@app.get("/api/corpora", response_model=List[CorpusResponse])
async def list_corpora():
    try:
        corpora = rag_agent.list_corpora()
        return [
            CorpusResponse(
                id=corpus.get("id", corpus.get("corpus_id", "")),
                name=corpus.get("name", ""),
                description=corpus.get("description"),
                document_count=corpus.get("document_count", 0),
                created_at=corpus.get("created_at"),
                updated_at=corpus.get("updated_at")
            )
            for corpus in corpora
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/corpora", response_model=CorpusResponse)
async def create_corpus(corpus: CorpusCreate):
    try:
        result = rag_agent.create_corpus(corpus.name, corpus.description)
        return CorpusResponse(
            id=result.get("corpus_id", ""),
            name=corpus.name,
            description=corpus.description,
            document_count=0,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/corpora/{corpus_id}")
async def delete_corpus(corpus_id: str):
    try:
        rag_agent.delete_corpus(corpus_id)
        return {"message": "Corpus deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/corpora/{corpus_id}", response_model=CorpusResponse)
async def get_corpus_info(corpus_id: str):
    try:
        info = rag_agent.get_corpus_info(corpus_id)
        return CorpusResponse(
            id=corpus_id,
            name=info.get("name", ""),
            description=info.get("description"),
            document_count=info.get("document_count", 0),
            created_at=info.get("created_at"),
            updated_at=info.get("updated_at")
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    corpus_id: str = Form(...)
):
    try:
        content = await file.read()
        
        if file.content_type in ['application/json']:
            data = json.loads(content)
        else:
            data = content.decode('utf-8')
        
        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "uploaded_at": datetime.now().isoformat()
        }
        
        result = rag_agent.add_data(
            corpus_id=corpus_id,
            data=data,
            metadata=metadata
        )
        
        return {"message": "Document uploaded successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    try:
        rag_agent.delete_document(document_id)
        return {"message": "Document deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        result = rag_agent.rag_query(
            corpus_id=request.corpus_id,
            query=request.query,
            num_results=request.num_results
        )
        
        sources = []
        if "sources" in result:
            for source in result["sources"]:
                sources.append({
                    "text": source.get("text", ""),
                    "metadata": source.get("metadata", {}),
                    "score": source.get("score", 0.0)
                })
        
        return QueryResponse(
            answer=result.get("answer", "No answer found"),
            sources=sources,
            confidence=result.get("confidence", 0.0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)