import os
import json
import time
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from . import models

from src.router import route

load_dotenv()
app = FastAPI(
    title="visdai",
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(route.router)


# COLLECTION_NAME="MovieDB"
# DB_USERNAME="postgres"
# DB_PASS="root"
# DB_HOST="localhost"
# DB_PORT=5432
# DB_DATABASE_NAME="vs_db"