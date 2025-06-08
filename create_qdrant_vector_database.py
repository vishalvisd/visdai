from qdrant_client import QdrantClient, models

def get_vd_client():
    client = QdrantClient(":memory:")

    client.create_collection(
      collection_name="MovieDB",
      vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

    return client