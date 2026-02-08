"""
Memory Manager for conversational SQL Assistant.
Manages short-term (ChromaDB) and long-term (PostgreSQL) conversation memory.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import chromadb
from chromadb.config import Settings
from src.utils import get_schema_prefix

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages conversation memory with ChromaDB (short-term) and PostgreSQL (long-term).
    """
    
    def __init__(self, user_id: int, session_id: str, db_manager, chroma_path: str = "./chroma_db"):
        """
        Initialize Memory Manager.
        
        Args:
            user_id: User ID from user_profiles table
            session_id: Current chat session ID
            db_manager: DatabaseManager instance for PostgreSQL operations
            chroma_path: Path to ChromaDB storage directory
        """
        self.user_id = user_id
        self.session_id = session_id
        self.db_manager = db_manager
        self.chroma_path = chroma_path
        
        # Initialize ChromaDB client
        self._init_chromadb()
        
        logger.info(f"MemoryManager initialized for user_id={user_id}, session_id={session_id}")
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection for short-term memory."""
        try:
            self.chroma_client = chromadb.Client(Settings(
                persist_directory=self.chroma_path,
                anonymized_telemetry=False
            ))
            
            # Create or get collection for this session
            collection_name = f"session_{self.session_id}"
            # ChromaDB collection names must be 3-63 characters and alphanumeric with underscores
            # Truncate if needed
            if len(collection_name) > 63:
                collection_name = collection_name[:63]
            
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"user_id": str(self.user_id), "session_id": self.session_id}
            )
            
            logger.info(f"ChromaDB collection initialized: {collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            self.collection = None
    
    def add_interaction(self, user_query: str, generated_sql: str = None, 
                       execution_result: Dict = None, feedback: str = None, 
                       feedback_type: str = None) -> bool:
        """
        Add a conversation interaction to both short-term and long-term memory.
        
        Args:
            user_query: User's natural language query
            generated_sql: Generated SQL query
            execution_result: Query execution result (dict with data)
            feedback: User feedback text
            feedback_type: Type of feedback ('positive', 'negative', 'correction')
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # 1. Add to ChromaDB (short-term memory)
            if self.collection:
                doc_id = f"turn_{datetime.now().timestamp()}"
                document = f"Query: {user_query}\nSQL: {generated_sql or 'N/A'}"
                if feedback:
                    document += f"\nFeedback: {feedback}"
                
                self.collection.add(
                    documents=[document],
                    ids=[doc_id],
                    metadatas=[{
                        "user_query": user_query,
                        "generated_sql": generated_sql or "",
                        "feedback": feedback or "",
                        "timestamp": str(datetime.now())
                    }]
                )
            
            # 2. Add to PostgreSQL (long-term memory)
            schema_prefix = get_schema_prefix(self.db_manager)
            table_name = f"{schema_prefix}conversation_history"
            
            insert_sql = f"""
            INSERT INTO {table_name} 
                (session_id, user_id, user_query, generated_sql, execution_result, feedback, feedback_type)
            VALUES 
                (:session_id, :user_id, :user_query, :generated_sql, :execution_result, :feedback, :feedback_type);
            """
            
            # Convert execution_result to JSON string for JSONB column
            result_json = json.dumps(execution_result) if execution_result else None
            
            self.db_manager.execute_query(insert_sql, params={
                "session_id": self.session_id,
                "user_id": self.user_id,
                "user_query": user_query,
                "generated_sql": generated_sql,
                "execution_result": result_json,
                "feedback": feedback,
                "feedback_type": feedback_type
            })
            
            logger.info(f"Added interaction to memory for session {self.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding interaction to memory: {str(e)}")
            return False
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for the current session from PostgreSQL.
        
        Args:
            limit: Maximum number of recent interactions to retrieve
            
        Returns:
            List of conversation turns as dictionaries
        """
        try:
            schema_prefix = get_schema_prefix(self.db_manager)
            table_name = f"{schema_prefix}conversation_history"
            
            query = f"""
            SELECT 
                user_query, 
                generated_sql, 
                execution_result, 
                feedback, 
                feedback_type,
                created_at
            FROM {table_name}
            WHERE session_id = :session_id
            ORDER BY created_at DESC
            LIMIT :limit;
            """
            
            df = self.db_manager.execute_query(query, params={
                "session_id": self.session_id,
                "limit": limit
            })
            
            if df.empty:
                return []
            
            # Convert to list of dictionaries (reverse to get chronological order)
            history = df.to_dict('records')
            history.reverse()  # Oldest first
            
            # Parse JSON execution results
            for turn in history:
                if turn.get('execution_result'):
                    try:
                        turn['execution_result'] = json.loads(turn['execution_result'])
                    except:
                        pass
            
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []
    
    def get_last_sql(self) -> Optional[str]:
        """
        Retrieve the most recent SQL query generated in this session.
        
        Returns:
            Last generated SQL or None if no SQL exists
        """
        try:
            schema_prefix = get_schema_prefix(self.db_manager)
            table_name = f"{schema_prefix}conversation_history"
            
            query = f"""
            SELECT generated_sql
            FROM {table_name}
            WHERE session_id = :session_id 
              AND generated_sql IS NOT NULL
              AND generated_sql != ''
            ORDER BY created_at DESC
            LIMIT 1;
            """
            
            df = self.db_manager.execute_query(query, params={
                "session_id": self.session_id
            })
            
            if not df.empty:
                last_sql = df.iloc[0]['generated_sql']
                logger.info(f"Retrieved last SQL for session {self.session_id}: {last_sql[:50]}...")
                return last_sql
            
            logger.info(f"No previous SQL found for session {self.session_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving last SQL: {str(e)}")
            return None
    
    def get_user_history(self, days_back: int = 7, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve user's conversation history across all sessions from PostgreSQL.
        
        Args:
            days_back: Number of days to look back
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of conversation turns from all user's sessions
        """
        try:
            schema_prefix = get_schema_prefix(self.db_manager)
            table_name = f"{schema_prefix}conversation_history"
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            query = f"""
            SELECT 
                session_id,
                user_query, 
                generated_sql, 
                feedback, 
                feedback_type,
                created_at
            FROM {table_name}
            WHERE user_id = :user_id 
              AND created_at >= :cutoff_date
            ORDER BY created_at DESC
            LIMIT :limit;
            """
            
            df = self.db_manager.execute_query(query, params={
                "user_id": self.user_id,
                "cutoff_date": cutoff_date,
                "limit": limit
            })
            
            if df.empty:
                return []
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error retrieving user history: {str(e)}")
            return []
    
    def format_for_prompt(self, limit: int = 10) -> str:
        """
        Format conversation history for inclusion in prompts.
        
        Args:
            limit: Maximum number of recent turns to include
            
        Returns:
            Formatted string representation of conversation history
        """
        history = self.get_conversation_history(limit=limit)
        
        if not history:
            return "No previous conversation in this session."
        
        formatted = []
        for i, turn in enumerate(history, 1):
            formatted.append(f"Turn {i}:")
            formatted.append(f"  User: {turn.get('user_query', 'N/A')}")
            formatted.append(f"  SQL: {turn.get('generated_sql', 'N/A')}")
            if turn.get('feedback'):
                formatted.append(f"  Feedback: {turn.get('feedback')} ({turn.get('feedback_type', 'general')})")
            formatted.append("---")
        
        return "\n".join(formatted)
    
    def clear_session(self) -> bool:
        """
        Clear short-term memory for the current session (ChromaDB only).
        Long-term PostgreSQL data is preserved.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.collection:
                # Delete all documents in the collection
                self.chroma_client.delete_collection(self.collection.name)
                # Recreate it
                self._init_chromadb()
                logger.info(f"Cleared session memory for {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing session: {str(e)}")
            return False
    
    def search_similar_interactions(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar past interactions using ChromaDB semantic search.
        
        Args:
            query: Query text to search for
            n_results: Number of similar results to return
            
        Returns:
            List of similar interactions
        """
        try:
            if not self.collection:
                return []
            
            # Query ChromaDB for similar documents
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if not results or not results['documents']:
                return []
            
            # Format results
            similar = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                similar.append({
                    "document": doc,
                    "metadata": metadata,
                    "distance": results['distances'][0][i] if results.get('distances') else None
                })
            
            return similar
            
        except Exception as e:
            logger.error(f"Error searching similar interactions: {str(e)}")
            return []
