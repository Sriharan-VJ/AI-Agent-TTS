from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .settings import ALLOWED_ORIGIN


def configure_cors(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[ALLOWED_ORIGIN],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
