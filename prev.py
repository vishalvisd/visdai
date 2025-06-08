template = """
You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".


Use the following format:


Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here


Only use the following tables:
{table_info}


Question: {input}
"""
import os
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from dotenv import load_dotenv
from huggingface_hub import login

os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

load_dotenv()

username = "postgres"
password = "root"
host = "localhost"
port = 5432
mydatabase = "postgres"


pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{mydatabase}"
db = SQLDatabase.from_uri(pg_uri)
print(db.dialect)
print(db.get_usable_table_names())
print(db.table_info)

# hf_token = os.environ['HF_TOKEN']
# login(hf_token, add_to_git_credential=True)

model_name = "Qwen/CodeQwen1.5-7B-Chat" # "mistralai/Mistral-7B-v0.1" #Qwen/CodeQwen1.5-7B-Chat
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
max_new_tokens = 10
pipe = pipeline(
"text-generation",
model=model_name,
tokenizer = tokenizer,
torch_dtype=torch.bfloat16,
device_map="auto",
max_new_tokens= max_new_tokens,
)
llm = HuggingFacePipeline(pipeline=pipe)

db_chain = create_sql_query_chain(llm, db)

question = "Which are the movies released in 2021?"
response = db_chain.invoke({"question" : question})
print(response)
