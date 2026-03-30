import chromadb
import logging
from typing import List, Dict, Any, Tuple
import os
import requests
import time

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

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
            
            if HAS_CROSS_ENCODER:
                logger.info("Initializing CrossEncoder reranker...")
                # MiniLM offers a great balance of speed and ranking quality
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
            else:
                logger.warning("sentence_transformers not installed; reranking disabled.")
                self.reranker = None
                
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
        Retrieval using ChromaDB with optional Cross-Encoder Reranking and Relevance Filtering.
        Returns: 
           ddl_content_list, top_tables_list, relevance_scores_dict, query_time_ms
        """
        t0 = time.time()
        try:
            query_embedding = self._generate_embeddings([prompt])
            
            # Fetch more documents if we have a reranker
            fetch_k = top_k * 3 if self.reranker else top_k
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=fetch_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            query_time_ms = int((time.time() - t0) * 1000)
            
            if not results['documents'] or not results['documents'][0]:
                return [], [], {}, query_time_ms
                
            candidates: List[Dict[str, Any]] = []
            
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                dist = results['distances'][0][i] if 'distances' in results and results['distances'] else 0.0
                sim = max(0.0, 1.0 - dist)
                
                table_name = meta['table_name']
                ddl_content = meta.get('ddl_content', doc)
                candidates.append({
                    'table_name': table_name,
                    'ddl_content': ddl_content,
                    'dense_score': sim,
                    # Fallback text to rank against
                    'rank_text': f"{table_name}: {ddl_content}" 
                })
                
            if self.reranker:
                # Prepare pairs for cross encoder: (query, document)
                pairs = [[prompt, c['rank_text']] for c in candidates]
                rerank_scores = self.reranker.predict(pairs)
                
                for idx, c in enumerate(candidates):
                    c['rerank_score'] = float(rerank_scores[idx])
                
                # RELEVANCE FILTERING: drop fundamentally irrelevant tables.
                # MS-MARCO MiniLM produces logits roughly between -10 and 10.
                # A score < -5.0 usually indicates zero semantic connection.
                relevance_threshold = -5.0
                filtered_candidates = [c for c in candidates if c['rerank_score'] >= relevance_threshold]
                
                # If everything was filtered out, fallback to dense top 1 (better than answering blindly)
                if not filtered_candidates and candidates:
                    logger.warning(f"All {len(candidates)} candidates filtered out by reranker. Keeping top 1 dense.")
                    candidates.sort(key=lambda x: x['dense_score'], reverse=True)
                    filtered_candidates = [candidates[0]]
                
                # Sort by rerank score descending
                filtered_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
                
                # Keep top K
                final_candidates = filtered_candidates[:top_k]
            else:
                # Dense only
                final_candidates = candidates[:top_k]
                
            ddls = []
            tables = []
            scores = {}
            for c in final_candidates:
                tables.append(c['table_name'])
                ddls.append(c['ddl_content'])
                # Report either the rerank score or dense score as similarity
                scores[c['table_name']] = round(c.get('rerank_score', c['dense_score']), 2)

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

