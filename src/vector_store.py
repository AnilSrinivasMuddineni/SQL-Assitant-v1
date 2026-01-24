import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any
import os

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name: str = "sql_assistant_ddls"):
        """
        Initialize the Vector Store using ChromaDB.
        """
        try:
            # Use a persistent client to save embeddings to disk
            # Ensure absolute path to avoid issues with CWD
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            chroma_path = os.path.join(project_root, "chroma_db")
            self.client = chromadb.PersistentClient(path=chroma_path)
            
            # Use a lightweight local model for embeddings
            # We wrap it in a custom embedding function for Chroma
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"} # Use cosine similarity
            )
            logger.info(f"Vector Store initialized with collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Vector Store: {str(e)}")
            raise e

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        return self.embedding_model.encode(texts).tolist()

    def store_ddls(self, ddls: Dict[str, str]):
        """
        Store DDLs in the vector database.
        
        Args:
            ddls: Dictionary mapping table names to their CREATE TABLE statements.
        """
        try:
            if not ddls:
                logger.warning("No DDLs provided to store.")
                return

            ids = list(ddls.keys())
            documents = list(ddls.values())
            
            # Generate embeddings
            embeddings = self._generate_embeddings(documents)
            
            # Upsert into ChromaDB
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=[{"table_name": name} for name in ids]
            )
            logger.info(f"Stored {len(ddls)} DDLs in Vector Store.")
            
        except Exception as e:
            logger.error(f"Error storing DDLs: {str(e)}")
            raise e

    def retrieve_relevant_ddls(self, query: str, n_results: int = 5) -> List[str]:
        """
        Retrieve the most relevant DDLs for a given query.
        
        Args:
            query: The user's natural language question.
            n_results: Number of relevant tables to retrieve.
            
        Returns:
            List of DDL strings.
        """
        try:
            query_embedding = self._generate_embeddings([query])
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            
            # Chroma returns a list of lists (one for each query)
            if results['documents'] and len(results['documents']) > 0:
                return results['documents'][0]
            return []
            
        except Exception as e:
            logger.error(f"Error retrieving DDLs: {str(e)}")
            return []

    def clear_store(self):
        """Clear all data from the collection."""
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(self.collection.name)
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
