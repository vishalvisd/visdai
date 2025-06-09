import os
from qdrant_client import models, QdrantClient

client = QdrantClient(url="http://localhost:6333")
collection_name = os.environ["COLLECTION_NAME"]
if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

