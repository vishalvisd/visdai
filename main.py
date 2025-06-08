from langchain_community.utilities import SQLDatabase
from transformers import AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Annotated, TypedDict

import torch

from schema import State

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# print(db.dialect)
# print(db.get_usable_table_names())
# print(db.run("SELECT * FROM Artist LIMIT 10;"))


model_name = "Qwen/CodeQwen1.5-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
max_new_tokens = 128

pipe = pipeline(
"text-generation",
model=model_name,
tokenizer = tokenizer,
torch_dtype=torch.bfloat16,
device_map="auto",
max_new_tokens= max_new_tokens,
)

llm = HuggingFacePipeline(pipeline=pipe)

system_message = """
You are a data assistant. Given a natural language question and database schema, your job is to output a valid {dialect} SQL query that answers the question.
Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Example:
Question: How many customers are there?
Answer: SELECT COUNT(*) FROM Customer;

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

⚠️ Do not explain anything. Do not confirm or ask questions. Do not say "let me think" or describe your reasoning. Just output the SQL query directly.

Only output the SQL query, and nothing else.

You must always return a valid SQL query that uses only the visible columns and tables from the schema below. Only use the following tables:

Schema:
{table_info}
"""

user_prompt = "Question: {input}\nAnswer:"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

for message in query_prompt_template.messages:
    message.pretty_print()


class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State):
    prompt_str = query_prompt_template.format(
        dialect=db.dialect,
        top_k=10,
        table_info=db.get_table_info(),
        input=state["question"],
    )

    print("prompt_str: ", prompt_str)
    result = llm.invoke(prompt_str)  # now a string, not a chat message list

    print("Model Output:", result)

    return {"query": result}  # or parse result if needed

# def write_query(state: State):
#     """Generate SQL query to fetch information."""
#     prompt = query_prompt_template.invoke(
#         {
#             "dialect": db.dialect,
#             "top_k": 10,
#             "table_info": db.get_table_info(),
#             "input": state["question"],
#         }
#     )
#
#     prompt_text = prompt.to_string()
#     result = llm.invoke(prompt_text)
#     return {"query": result["query"]}


print(write_query({"question": "How many Employees are there?"}))
