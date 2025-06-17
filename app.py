import os
import logging
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_pagination import add_pagination

from configurations.config_reader import read_config
from configurations.params import LOGGING_FILE_PATH
from routers.auth import secure
from routers.email_adaptor import email_classification
from source.faiss_client import faiss_save_all

logger = logging.getLogger(__name__)
logging.basicConfig(filename=LOGGING_FILE_PATH, level=logging.INFO)
config = read_config("config.ini")
ENV = os.getenv("ENV_TYPE", "DEV")

@asynccontextmanager
async def lifespan(_: FastAPI):
    load_dotenv()
    
    # load or create the faiss db on startup
    # vector_store = faiss_index(faiss_index_path=FAISS_DB_PATH)

    yield

    # clean up
    faiss_save_all()


app = FastAPI(
    lifespan=lifespan,
    title="KYC-BOT-EMAIL-PROCESSING",
    description="Classify and extract entities from emails",
    summary="",
    version="0.0.1",
    contact={
        "name": "Madhavi Mehta"
    }
)

add_pagination(app)
app.include_router(email_classification)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host = config[ENV]["host"],
        port = int(config[ENV]["port"]),
        log_level="debug",
        workers=int(config[ENV]["workers"]),
        reload=False,
        ssl_keyfile=config[ENV]["ssl_keyfile"],
        ssl_certfile=config[ENV]["ssl_certfile"],
    )