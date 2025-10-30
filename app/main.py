from fastapi import FastAPI
from .api.routes import router
from .core.model import model_bundle
from .core.config import settings

app = FastAPI(title=settings.title, version=settings.version,
              description="XLM-R toxicity detection API")


@app.on_event("startup")
def _startup() -> None:
    model_bundle.load()


app.include_router(router)
