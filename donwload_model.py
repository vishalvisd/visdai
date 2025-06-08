from huggingface_hub import snapshot_download

# Just download the model into the default Hugging Face cache
snapshot_download(
    repo_id="Qwen/CodeQwen1.5-7B-Chat", #Qwen/CodeQwen1.5-7B-Chat #mistralai/Mistral-7B-Instruct-v0.1
    local_dir=None,                # Use default cache location
    local_dir_use_symlinks=True,  # Optional: saves space
)



