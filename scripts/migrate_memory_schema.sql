-- =============================================================================
-- Memory Schema Migration: Stateful Conversational SQL Assistant
-- Database: Custom_RAG  |  Schema: sql_rag
-- Run once (idempotent - safe to re-run)
-- Requires: PostgreSQL 12+ (for GENERATED ALWAYS AS STORED on TSVECTOR)
-- =============================================================================

-- Ensure schema exists
CREATE SCHEMA IF NOT EXISTS sql_rag;

-- =============================================================================
-- Table 1: chat_sessions
-- One row per user login / "Start Session" click.
-- user_id uses normalized string format: "emp:12345" or "mob:9876543210"
-- =============================================================================
CREATE TABLE IF NOT EXISTS sql_rag.chat_sessions (
    session_id      TEXT        PRIMARY KEY,
    user_id         TEXT        NOT NULL,
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    is_active       BOOLEAN     DEFAULT TRUE,
    metadata        JSONB       DEFAULT '{}'::JSONB
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id
    ON sql_rag.chat_sessions(user_id);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_active
    ON sql_rag.chat_sessions(user_id, is_active)
    WHERE is_active = TRUE;

-- =============================================================================
-- Table 2: chat_messages
-- One row per message turn (role: user | assistant | sql).
-- search_tsv is auto-generated from content for full-text search.
-- sql_generated stores parsed SQL block as structured JSONB.
-- =============================================================================
CREATE TABLE IF NOT EXISTS sql_rag.chat_messages (
    id                  BIGSERIAL   PRIMARY KEY,
    user_id             TEXT        NOT NULL,
    session_id          TEXT        NOT NULL
                            REFERENCES sql_rag.chat_sessions(session_id)
                            ON DELETE CASCADE,
    role                TEXT        NOT NULL
                            CHECK (role IN ('user', 'assistant', 'sql')),
    content             TEXT        NOT NULL,
    sql_generated       JSONB,          -- {"query": "SELECT ...", "tables": [...]}
    feedback_summary    TEXT,           -- extracted modification keywords
    search_tsv          TSVECTOR
                            GENERATED ALWAYS AS
                            (to_tsvector('english', content)) STORED,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS idx_chat_messages_tsv
    ON sql_rag.chat_messages USING GIN(search_tsv);

-- Composite index for per-user chronological retrieval
CREATE INDEX IF NOT EXISTS idx_chat_messages_user_created
    ON sql_rag.chat_messages(user_id, created_at DESC);

-- Index for session-level queries
CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON sql_rag.chat_messages(session_id, created_at DESC);

-- Partial index for SQL role messages (for fast last-SQL lookups)
CREATE INDEX IF NOT EXISTS idx_chat_messages_user_sql
    ON sql_rag.chat_messages(user_id, created_at DESC)
    WHERE role = 'sql';

-- =============================================================================
-- Verify tables created successfully
-- =============================================================================
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'sql_rag'
          AND table_name = 'chat_sessions'
    ) AND EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'sql_rag'
          AND table_name = 'chat_messages'
    ) THEN
        RAISE NOTICE 'Migration successful: chat_sessions and chat_messages tables ready.';
    ELSE
        RAISE EXCEPTION 'Migration FAILED: One or more tables not created.';
    END IF;
END;
$$;
