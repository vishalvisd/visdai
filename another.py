from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.llms import HuggingFacePipeline
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from qdrant_client import models, QdrantClient
import uuid

# question = "Which director has maximum movies?"
question = "Which are the movies whose name starts with 'b'?"
# question = "total length of all movies?"
# question = "Which movies released in 2019?"

#nyOFoGpCrgRjFrAxyPBgswZoIRAxACKVHu
client = QdrantClient(url="http://localhost:6333")

collection_name = "MovieDB"

if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim vectors


VECTOR_SEARCH_THRESHOLD = 0.9  # Adjust as needed

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

def store_question_sql_pair(question: str, sql_query: str):
    print("Saving into vector db. Question:", question, ", SQL:", sql_query)
    vector = get_question_vector(question)
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),  # unique ID for each entry
                vector=vector,
                payload={"question": question, "sql_query": sql_query},
            )
        ]
    )

# Step 1: Try vector search first
cached_sql = search_similar_question(question)

sql_query = None
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

    username = "postgres"
    password = "root"
    host = "localhost"
    port = 5432
    mydatabase = "vs_db"

    pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{mydatabase}"
    db = SQLDatabase.from_uri(pg_uri)

    model_name = "Qwen/CodeQwen1.5-7B-Chat"
    # model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    max_new_tokens = 256

    pipe = pipeline(
      "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float32,  # for CPU
        max_new_tokens=1024,
        return_full_text=False,
    )
    llm = HuggingFacePipeline(pipeline=pipe )

    # ---- RAG Application Flow ----
    table_info = db.get_table_info()
    prompt = format_prompt(model_name, table_info, question) # template.format(table_info=table_info, input=question)
    print("Input Prompt:\n", prompt)
    raw_output = llm.invoke(prompt)
    print("Raw Output:\n", raw_output)

    # Step 2: Extract SQL query from output
    def extract_sql(text):
        if "SQLQuery:" in text:
            return text.split("SQLQuery:")[-1].split("\n")[0].strip()
        return text.strip()

    sql_query = extract_sql(raw_output)
    print("\nExtracted SQL Query:\n", sql_query)
    store_question_sql_pair(question, sql_query)

if sql_query:
    print("Answer: ", sql_query)
else:
    print("Failed to generate query")


# points = []
# point = models.PointStruct(
#   id=i,
#   payload={
#     "natural_lan_query": question,
#     "sql_query": response
#   },
#   vector=vector,
# )
# points.append(point)
# client.upsert(
#   collection_name="MovieDB",
#   points=points,
# )
#
#
# encoder = SentenceTransformer('BAAI/bge-small-en')
# hits = client.search(
#   collection_name="MovieDB",
#   query_vector=encoder.encode(question).tolist(),
#   limit=1,
# )
# if hits[0].score > 0.90:
#   print(hits[0].payload)
# else:
#   response = db_chain.invoke({"question" : question})
#   print(response)

#
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain.llms import HuggingFacePipeline
# from langchain_community.utilities import SQLDatabase
# from langchain.chains import create_sql_query_chain
# import torch
# # Load tokenizer and model from local cache (if already downloaded)
# model_name = "Qwen/CodeQwen1.5-7B-Chat"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
#
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.float32,  # for CPU
#     max_new_tokens=128,
#     return_full_text=False,
# )
#
# llm = HuggingFacePipeline(pipeline=pipe)
#
# # Prompt template
# template = """
# You are a PostgreSQL expert. Given an input question, generate a syntactically correct PostgreSQL query that can answer the question.
# Do NOT attempt to run the query or guess the result. Just return the SQL query needed to answer the question.
#
# Rules:
# - Use LIMIT 5 if the question does not specify number of rows
# - Never SELECT * — only select the required columns
# - Wrap column names in double quotes (")
# - Use only the columns and tables shown below
# - If the question says "today", use date('now')
#
# Schema:
# CREATE TABLE movies (
# 	id SERIAL NOT NULL,
# 	title VARCHAR(255) NOT NULL,
# 	release_year INTEGER,
# 	genre VARCHAR(100),
# 	director VARCHAR(255),
# 	duration_minutes INTEGER,
# 	CONSTRAINT movies_pkey PRIMARY KEY (id)
# )
#
# Question: Which are the movies released in 2021?
# SQLQuery:
# """
#
# # Step 1: Generate from model
# raw_output = llm.invoke(template)
# print("Raw Output:\n", raw_output)
#
# # Step 2: Extract SQL query from output
# def extract_sql(text):
#     if "SQLQuery:" in text:
#         return text.split("SQLQuery:")[-1].split("\n")[0].strip()
#     return text.strip()
#
# sql_query = extract_sql(raw_output)
# print("\nExtracted SQL Query:\n", sql_query)