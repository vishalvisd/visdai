import os
import json
from typing import List
from datetime import datetime, timezone
import threading
import uuid

from fastapi import APIRouter, Depends, Response, status, HTTPException
from pydantic.v1.schema import json_scheme
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
from sentence_transformers import SentenceTransformer
from langchain_community.utilities import SQLDatabase
from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from langchain.chains import create_sql_query_chain
from langchain.llms import HuggingFacePipeline
from qdrant_client import models as qdrant_models
import torch

from .. import models, schemas
from ..vectordb import client, collection_name

router = APIRouter(
    prefix="/v1",
    tags=['v1'],
)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
VECTOR_SEARCH_THRESHOLD = 0.9

def get_question_vector(question: str):
    return embedding_model.encode(question).tolist()

def search_similar_question(question: str):
    vector = get_question_vector(question)
    search_result = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=1,
        score_threshold=VECTOR_SEARCH_THRESHOLD,
    )
    if search_result:
        return search_result[0].payload["sql_query"]
    return None

def store_question_sql_pair(question: str, query: str):
    print("Saving into vector db. Question:", question, ", SQL:", query)
    vector = get_question_vector(question)
    client.upsert(
        collection_name=collection_name,
        points=[
            qdrant_models.PointStruct(
                id=str(uuid.uuid4()),  # unique ID for each entry
                vector=vector,
                payload={"question": question, "sql_query": query},
            )
        ]
    )
sql_query = None

@router.get("/answer")
def answer(
    request: schemas.FindAnswer,
):
    question = request.question
    cached_sql = search_similar_question(question)
    username = os.environ.get("DB_USERNAME")
    password = os.environ.get("DB_PASS")
    host = os.environ.get("DB_HOST")
    port = os.environ.get("DB_PORT")
    mydatabase = os.environ.get("DB_DATABASE_NAME")

    pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{mydatabase}"
    db = SQLDatabase.from_uri(pg_uri)

    global sql_query
    if cached_sql:
        print("\nFound similar question in Qdrant!")
        sql_query = cached_sql
    else:
        print("\nNo match found, using LLM to generate SQL...")

        def format_prompt(model_name: str, table_info: str, question: str) -> str:
            # Core reusable components
            system_message = """
            You are a PostgreSQL expert.

            Given the following database schema and a user question, return only one SQL query that answers it. Do not return any explanation, follow-up questions, or additional queries.

            Rules:
            - Output only a single SQL statement.
            - Do not include the original question or any other text.
            - Wrap column names in double quotes (").
            - Use LIMIT 5 unless otherwise stated.
            - Use only the schema shown below.
            - Use date('now') for "today".
            """.strip()

            user_message = f"""
            Schema:
            {table_info}

            Question: {question}
            SQL:
            """.strip()

            # Normalize model name
            model_name = model_name.lower()

            # Chat-style formatting for chat-tuned models
            if any(chat_model in model_name for chat_model in ["qwen", "chatglm", "baichuan", "openchat"]):
                return f"""
        <|im_start|>system
        {system_message}
        <|im_end|>

        <|im_start|>user
        {user_message}
        <|im_end|>

        <|im_start|>assistant
        """.strip()

            # Instruction-style for models like Mistral
            return f"{system_message}\n\n{user_message}"

        model_name = "Qwen/CodeQwen1.5-7B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        max_new_tokens = 1024

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float32,  # for CPU
            max_new_tokens=1024,
            return_full_text=False,
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        table_info = db.get_table_info()
        prompt = format_prompt(model_name, table_info, question)

        print("Input Prompt:\n", prompt)
        raw_output = llm.invoke(prompt)
        print("Raw Output:\n", raw_output)

        def extract_sql(text):
            if "SQLQuery:" in text:
                return text.split("SQLQuery:")[-1].split("\n")[0].strip()
            return text.strip()

        sql_query = extract_sql(raw_output)

    if sql_query:
        db_result = db.run(sql_query)
        return {
            "data": db_result
        }

    else:
        print("Failed to generate query")
        return {
            "error": True
        }
