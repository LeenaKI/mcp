import logging
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

Metadata = Dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Optional[Metadata] = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the collection to use.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: Optional[str],
        qdrant_api_key: Optional[str],
        collection_name: str,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: Optional[str] = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )

    async def _ensure_collection_exists(self):
        """Ensure that the collection exists, creating it if necessary with binary quantization."""
        collection_exists = await self._client.collection_exists(self._collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            sample_vector = await self._embedding_provider.embed_query("sample text")
            vector_size = len(sample_vector)

            # Use the vector name as defined in the embedding provider
            vector_name = self._embedding_provider.get_vector_name()
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,  # Use Cosine for float vectors
                        on_disk=True,  # Store full vectors on disk to save memory
                    )
                },
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True)  # Enable binary quantization
                ),
            )
            logger.info(f"Created collection {self._collection_name} with binary quantization")

    async def store(self, entry: Entry):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        Logs the input, original vectors, and approximated binary quantized vectors.
        """
        await self._ensure_collection_exists()
        logger.info(f"Storing input: {entry.content}")
        embeddings = await self._embedding_provider.embed_documents([entry.content])
        original_vector = embeddings[0]
        logger.info(f"Original vector (first 10 dims): {original_vector[:10]} (length: {len(original_vector)})")
        binary_vector = [1 if x > 0 else 0 for x in original_vector]
        logger.info(f"Approximated binary vector (first 10 dims): {binary_vector[:10]} (length: {len(binary_vector)})")
        vector_name = self._embedding_provider.get_vector_name()
        payload = {"document": entry.content, "metadata": entry.metadata}
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={vector_name: original_vector},
                    payload=payload,
                )
            ],
        )
        logger.info(f"Successfully stored entry in Qdrant: {entry.content}")

    async def search(self, query: str) -> list[Entry]:
        """
        Find points in the Qdrant collection with relevance filtering.
        :param query: The query to use for the search.
        :return: A list of relevant entries found.
        """
        logger.info(f"Searching for query: '{query}'")
        collection_exists = await self._client.collection_exists(self._collection_name)
        if not collection_exists:
            logger.warning(f"Collection {self._collection_name} does not exist")
            return []

        # Embed the query
        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()
        logger.info(f"Query vector (first 10 dims): {query_vector[:10]} (length: {len(query_vector)})")

        # Search in Qdrant with quantization and rescoring
        search_results = await self._client.search(
            collection_name=self._collection_name,
            query_vector=models.NamedVector(name=vector_name, vector=query_vector),
            limit=10,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=True,  # Refine with full vectors
                    oversampling=2.0,  # Increase candidate pool
                )
            ),
        )

        # Filter results by similarity score
        logger.info(f"Found {len(search_results)} initial results")
        relevant_entries = []
        for result in search_results:
            score = result.score
            content = result.payload["document"]
            logger.info(f"Result: Score={score:.3f}, Content='{content}'")
            if score > 0.5:  # Adjustable threshold
                relevant_entries.append(
                    Entry(
                        content=content,
                        metadata=result.payload.get("metadata"),
                    )
                )

        if not relevant_entries:
            logger.info("No results above similarity threshold 0.5")
            return [Entry(content="No relevant results found", metadata=None)]
        logger.info(f"Returning {len(relevant_entries)} relevant results")
        return relevant_entries