-- Logging tables for Observability Dashboard and semantic DDL retrieval
-- These tables are used by the AgentLogger class in src/logger.py

-- Table 1: agent_request_response
CREATE TABLE IF NOT EXISTS agent_request_response (
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
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table 2: ddl_retrieval_log
CREATE TABLE IF NOT EXISTS ddl_retrieval_log (
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

-- Indices for fast retrieval of performance metrics
CREATE INDEX IF NOT EXISTS idx_agent_user_time ON agent_request_response(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ddl_user_time ON ddl_retrieval_log(user_id, created_at DESC);
