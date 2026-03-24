"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import HealthResponse
from src.api.routes import images, search
from src.retrieval.search import Searcher


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    app.state.config = config

    print("Loading searcher (CLIP + FAISS) ...")
    app.state.searcher = Searcher(config)

    # Optionally load rerankers if available
    try:
        from src.reranking.metadata_reranker import MetadataReranker
        app.state.metadata_reranker = MetadataReranker(config)
        print("MetadataReranker loaded.")
    except Exception as e:
        print(f"MetadataReranker not loaded: {e}")

    try:
        from src.reranking.multimodal_reranker import MultimodalReranker
        app.state.multimodal_reranker = MultimodalReranker(config, app.state.searcher.encoder)
        print("MultimodalReranker loaded.")
    except Exception as e:
        print(f"MultimodalReranker not loaded: {e}")

    print("API ready.")
    yield

    print("Shutting down ...")


app = FastAPI(
    title="Pinterest Visual Search API",
    description="Context-aware visual image search system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router, tags=["Search"])
app.include_router(images.router, tags=["Images"])


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health(request: "any" = None):
    from fastapi import Request
    state = app.state
    return HealthResponse(
        status="ok",
        model=state.config["model"]["clip_model"],
        index_size=state.searcher.index.index.ntotal if state.searcher.index.index else 0,
        device=state.searcher.encoder.device,
    )


if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    uvicorn.run(
        "src.api.main:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=cfg["api"]["reload"],
    )
