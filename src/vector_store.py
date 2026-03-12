import chromadb
import logging
from typing import List, Dict, Any, Tuple
import os
import requests
import time

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name: str = "sql_ddl_embeddings"):
        """
        Initialize the Vector Store using ChromaDB.
        """
        try:
            # Persistent client in ./chromadb_sql_ddl
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            chroma_path = os.path.join(project_root, "chromadb_sql_ddl")
            self.client = chromadb.PersistentClient(path=chroma_path)
            
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"} # Use cosine similarity
            )
            logger.info(f"Vector Store initialized with collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Vector Store: {str(e)}")
            raise e

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using nomic-embed-text via Ollama."""
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": text}
                )
                if response.status_code == 200:
                    embeddings.append(response.json()["embedding"])
                else:
                    logger.error(f"Ollama embedding error: {response.text}")
                    # Fallback padding if failed
                    embeddings.append([0.0]*768)
            except Exception as e:
                logger.error(f"Ollama connection error: {e}")
                embeddings.append([0.0]*768)
        return embeddings

    def store_ddls(self, ddls: Dict[str, str]):
        """
        Store DDLs in the vector database using batch embedding.
        
        Args:
            ddls: Dictionary mapping table names to their CREATE TABLE statements.
        """
        try:
            if not ddls:
                logger.warning("No DDLs provided to store.")
                return

            ids = list(ddls.keys())
            documents = list(ddls.values())
            
            # Formatted document for better semantic search:
            # "{table_name}: {ddl}"
            enhanced_documents = [f"{name}: {doc}" for name, doc in zip(ids, documents)]
            
            # Generate embeddings
            embeddings = self._generate_embeddings(enhanced_documents)
            
            # Upsert into ChromaDB
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=[{"table_name": name, "ddl_content": doc} for name, doc in zip(ids, documents)]
            )
            logger.info(f"Stored {len(ddls)} DDLs in Vector Store (ChromaDB + Nomic).")
            
        except Exception as e:
            logger.error(f"Error storing DDLs: {str(e)}")
            raise e

    def retrieve_relevant_ddls(self, query: str, n_results: int = 5) -> List[str]:
        # Keep original method for compatibility
        res = self.semantic_search_ddl(query, n_results)
        return res[0]

    def semantic_search_ddl(self, prompt: str, top_k: int = 5) -> Tuple[List[str], List[str], Dict[str, float], int]:
        """
        Performance-Optimized Retrieval using ChromaDB.
        Returns: 
           ddl_content_list, top_tables_list, relevance_scores_dict, query_time_ms
        """
        t0 = time.time()
        try:
            query_embedding = self._generate_embeddings([prompt])
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            query_time_ms = int((time.time() - t0) * 1000)
            
            ddls = []
            tables = []
            scores = {}
            
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    doc = results['documents'][0][i]
                    meta = results['metadatas'][0][i]
                    # We might not get distances if not returned, so handle gracefully
                    dist = results['distances'][0][i] if 'distances' in results and results['distances'] else 0.0
                    
                    # Cosine distance to similarity score
                    sim = max(0.0, 1.0 - dist)
                    
                    # Original doc is in result['documents'] or metadata['ddl_content']
                    table_name = meta['table_name']
                    # We stored original DDL in metadata to avoid the Prefix we added
                    ddl_content = meta.get('ddl_content', doc)
                    
                    tables.append(table_name)
                    ddls.append(ddl_content)
                    scores[table_name] = round(sim, 2)
                    
            return ddls, tables, scores, query_time_ms
            
        except Exception as e:
            logger.error(f"Error in semantic_search_ddl: {str(e)}")
            return [], [], {}, int((time.time() - t0) * 1000)

    def clear_store(self):
        """Clear all data from the collection."""
        try:
            existing = self.collection.get()
            if existing and existing['ids']:
                self.collection.delete(ids=existing['ids'])
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")

