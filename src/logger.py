import time
import json
import logging
from functools import wraps
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class AgentLogger:
    def __init__(self, db_manager):
        """Initialize AgentLogger with the Postgres DatabaseManager."""
        self.db_manager = db_manager
        
    def init_logging_tables(self):
        """Initialize the Postgres JSONB logging tables."""
        from src.utils import get_schema_prefix
        schema_prefix = get_schema_prefix(self.db_manager)
        
        # Table 1: agent_request_response
        create_agent_log_sql = f"""
        CREATE TABLE IF NOT EXISTS {schema_prefix}agent_request_response (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            message_id INTEGER,
            agent_type TEXT NOT NULL,
            request_prompt TEXT NOT NULL,
            response_output TEXT NOT NULL,
            time_taken_ms INTEGER NOT NULL,
            input_tokens INTEGER, 
            output_tokens INTEGER,
            sql_detected BOOLEAN DEFAULT FALSE,
            tables_used TEXT[],
            metadata JSONB DEFAULT '{{}}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        # Table 2: ddl_retrieval_log
        create_ddl_log_sql = f"""
        CREATE TABLE IF NOT EXISTS {schema_prefix}ddl_retrieval_log (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            message_id INTEGER,
            user_prompt TEXT NOT NULL,
            matched_tables TEXT[] NOT NULL,
            relevance_scores JSONB NOT NULL,
            top_k INTEGER DEFAULT 3,
            chroma_query_time_ms INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        try:
            self.db_manager.execute_query(create_agent_log_sql)
            self.db_manager.execute_query(create_ddl_log_sql)
            
            # Create indexes. Not using CONCURRENTLY to avoid transaction errors in SQLAlchemy without autocommit
            self.db_manager.execute_query(f"CREATE INDEX IF NOT EXISTS idx_agent_user_time ON {schema_prefix}agent_request_response(user_id, created_at DESC);")
            self.db_manager.execute_query(f"CREATE INDEX IF NOT EXISTS idx_ddl_user_time ON {schema_prefix}ddl_retrieval_log(user_id, created_at DESC);")
            
            logger.info("Logging tables and indexes verified.")
        except Exception as e:
            logger.error(f"Error creating logging tables: {e}")

    def log_semantic_ddl_retrieval(self, user_id: str, message_id: Optional[int], prompt: str, 
                                  top_tables: List[str], scores: Dict[str, float], query_time_ms: int):
        from src.utils import get_schema_prefix
        schema_prefix = get_schema_prefix(self.db_manager)
        try:
            sql = f"""
            INSERT INTO {schema_prefix}ddl_retrieval_log 
            (user_id, message_id, user_prompt, matched_tables, relevance_scores, top_k, chroma_query_time_ms)
            VALUES (:user_id, :message_id, :prompt, :tables, :scores, :top_k, :time_ms)
            """
            
            self.db_manager.execute_query(sql, params={
                "user_id": user_id,
                "message_id": message_id,
                "prompt": prompt,
                "tables": top_tables,
                "scores": json.dumps(scores),
                "top_k": len(top_tables),
                "time_ms": query_time_ms
            })
        except Exception as e:
            logger.error(f"Failed to log DDL retrieval: {e}")

    def log_agent_interaction(self, user_id: str, message_id: Optional[int], agent_type: str, 
                             request: str, response: str, time_ms: int, 
                             input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
                             tables_used: Optional[List[str]] = None, metadata: Optional[Dict] = None):
        from src.utils import get_schema_prefix
        schema_prefix = get_schema_prefix(self.db_manager)
        
        try:
            sql_detected = bool(response and any(verb in response.upper() for verb in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']))
            
            sql = f"""
            INSERT INTO {schema_prefix}agent_request_response 
            (user_id, message_id, agent_type, request_prompt, response_output, time_taken_ms, 
             input_tokens, output_tokens, sql_detected, tables_used, metadata)
            VALUES (:user_id, :message_id, :agent_type, :request, :response, :time_ms, 
                    :in_tok, :out_tok, :sql_det, :tables, :meta)
            """
            
            self.db_manager.execute_query(sql, params={
                "user_id": user_id,
                "message_id": message_id,
                "agent_type": agent_type,
                "request": request,
                "response": response,
                "time_ms": time_ms,
                "in_tok": input_tokens,
                "out_tok": output_tokens,
                "sql_det": sql_detected,
                "tables": tables_used,
                "meta": json.dumps(metadata) if metadata else '{}'
            })
        except Exception as e:
            logger.error(f"Failed to log agent interaction: {e}")

    def get_performance_dashboard(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        from src.utils import get_schema_prefix
        schema_prefix = get_schema_prefix(self.db_manager)
        result = {}
        
        try:
            # ChromaDB retrieval quality
            q_ddl = f"""
            SELECT relevance_scores, chroma_query_time_ms, matched_tables, user_prompt
            FROM {schema_prefix}ddl_retrieval_log 
            WHERE user_id = :user_id AND created_at >= NOW() - INTERVAL '{days} days'
            ORDER BY created_at DESC LIMIT 100
            """
            ddl_df = self.db_manager.execute_query(q_ddl, params={"user_id": user_id})
            
            avg_time = ddl_df['chroma_query_time_ms'].mean() if not ddl_df.empty else 0
            
            total_score = 0
            total_items = 0
            high_relevance_count = 0
            
            for _, row in ddl_df.iterrows():
                try:
                    scores = json.loads(row['relevance_scores']) if isinstance(row['relevance_scores'], str) else row['relevance_scores']
                    for s in scores.values():
                        total_score += s
                        total_items += 1
                        if s > 0.85:
                            high_relevance_count += 1
                except:
                    pass
                    
            avg_relevance = (total_score / total_items) if total_items > 0 else 0
            pct_high = (high_relevance_count / total_items) * 100 if total_items > 0 else 0
            
            result['chroma'] = {
                'avg_time_ms': round(avg_time, 2),
                'avg_relevance': round(avg_relevance, 2),
                'pct_high_relevance': round(pct_high, 1),
                'recent_logs': ddl_df.head(5).to_dict('records') if not ddl_df.empty else []
            }
            
            # Agent bottlenecks
            q_agent = f"""
            SELECT agent_type, 
                   COUNT(*) as calls,
                   AVG(time_taken_ms) as avg_time,
                   SUM(CASE WHEN sql_detected = TRUE THEN 1 ELSE 0 END) as sql_success_count
            FROM {schema_prefix}agent_request_response 
            WHERE user_id = :user_id AND created_at >= NOW() - INTERVAL '{days} days'
            GROUP BY agent_type
            ORDER BY avg_time DESC
            """
            agent_df = self.db_manager.execute_query(q_agent, params={"user_id": user_id})
            
            agent_stats = []
            total_sql_calls = 0
            total_sql_success = 0
            
            for _, r in agent_df.iterrows():
                sql_success_rate = (r['sql_success_count'] / r['calls']) * 100 if r['calls'] > 0 else 0
                agent_stats.append({
                    'agent': r['agent_type'],
                    'avg_time': round(r['avg_time'], 2),
                    'calls': r['calls'],
                    'sql_success_rate': round(sql_success_rate, 1)
                })
                if r['agent_type'] == 'sql_developer' or r['agent_type'] == 'developer':
                    total_sql_calls += r['calls']
                    total_sql_success += r['sql_success_count']
                    
            overall_sql_success = (total_sql_success / total_sql_calls) * 100 if total_sql_calls > 0 else 0
            
            result['agent_pipeline'] = agent_stats
            result['overall_sql_success'] = round(overall_sql_success, 1)
            
            # Raw logs
            q_raw = f"""
            SELECT agent_type, request_prompt, time_taken_ms, sql_detected, created_at
            FROM {schema_prefix}agent_request_response
            WHERE user_id = :user_id
            ORDER BY created_at DESC LIMIT 50
            """
            raw_df = self.db_manager.execute_query(q_raw, params={"user_id": user_id})
            result['raw_logs'] = raw_df.to_dict('records') if not raw_df.empty else []
            
            return result
        except Exception as e:
            logger.error(f"Error getting performance dashboard: {e}")
            return {}

def log_agent_step(agent_type: str):
    """
    Decorator to log agent latency and I/O.
    The decorated function should receive a kwargs 'agent_logger' and 'user_id' OR
    be a method on a class with 'logger' and 'string_user_id' properties.
    """
    def decorator(func):
        if hasattr(func, '_is_coroutine') or __import__('asyncio').iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                t0 = time.time()
                try:
                    res = await func(self, *args, **kwargs)
                    return res
                finally:
                    t_ms = int((time.time() - t0) * 1000)
                    agent_logger = getattr(self, 'logger', kwargs.get('agent_logger'))
                    user_id = getattr(self, 'string_user_id', kwargs.get('user_id', 'unknown'))
                    if agent_logger:
                        prompt = kwargs.get('query', args[0] if args else "unknown")
                        response_str = str(res) if res else ""
                        agent_logger.log_agent_interaction(user_id, None, agent_type, str(prompt), response_str, t_ms)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(self, *args, **kwargs):
                t0 = time.time()
                try:
                    res = func(self, *args, **kwargs)
                    return res
                finally:
                    t_ms = int((time.time() - t0) * 1000)
                    agent_logger = getattr(self, 'logger', kwargs.get('agent_logger'))
                    user_id = getattr(self, 'string_user_id', kwargs.get('user_id', 'unknown'))
                    if agent_logger:
                        prompt = kwargs.get('query', args[0] if args else "unknown")
                        response_str = str(res) if res else ""
                        agent_logger.log_agent_interaction(user_id, None, agent_type, str(prompt), response_str, t_ms)
            return sync_wrapper
    return decorator
