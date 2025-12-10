from fastapi import FastAPI
from retrieval import search
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Search API is running"}

@app.get("/")
async def root():
    return {"message": "Welcome to Search API", "docs": "/docs"}

@app.get("/search")
async def search_endpoint(query: str):
    results, resp_time = search(query)
    return {"query": query, "response_time": resp_time, "results": results}