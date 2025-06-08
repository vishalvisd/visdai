from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "Qwen/CodeQwen1.5-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    return_full_text=False
)

prompt = "Write a PostgreSQL query to get top 5 longest movies from a table named movies."
output = pipe(prompt)
print(output)
