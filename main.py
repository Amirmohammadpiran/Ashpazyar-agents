# app/main.py
from fastapi import FastAPI
from routers import smart_search, alternative_finder

app = FastAPI(title="Smart Multi-Agent Server")

app.include_router(smart_search.router)
app.include_router(alternative_finder.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8200, reload=True)
