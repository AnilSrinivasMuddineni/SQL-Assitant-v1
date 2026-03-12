"""
PostgresTextMemory: Per-user conversational memory for SQL Assistant.

Uses PostgreSQL TSVECTOR full-text search + recent-message retrieval
for hybrid context. No pgvector or external vector extensions required.

Schema: sql_rag.chat_sessions + sql_rag.chat_messages
User ID format: "emp:12345" or "mob:9876543210"
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from sqlalchemy import text

logger = logging.getLogger(__name__)


class PostgresTextMemory:
    """
    Per-user conversational memory backed by PostgreSQL text-search.

    Provides:
    - save_message()        : Persist each turn (user / assistant / sql roles)
    - get_recent_history()  : Chronological last-N messages per user
    - get_relevant_context(): Hybrid = recent + FTS ranked by current query terms
    - extract_feedback()    : Regex detection of modification intent
    - get_last_sql()        : Latest SQL block for a user
    - format_for_prompt()   : Render message list into prompt string
    """

    # -------------------------------------------------------------------------
    # Modification-intent keywords for extract_feedback()
    # -------------------------------------------------------------------------
    _ACTION_WORDS = re.compile(
        r'\b(fix|modify|change|update|correct|edit|alter|adjust|wrong|bad|incorrect|replace|rewrite)\b',
        re.IGNORECASE
    )
    _TARGET_WORDS = re.compile(
        r'\b(join|filter|where|group\s+by|groupby|having|order\s+by|orderby|'
        r'column|field|table|select|limit|aggregate|sum|count|avg|date|range|condition)\b',
        re.IGNORECASE
    )
    _REFERENCE_WORDS = re.compile(
        r'\b(previous|last|prior|above|that|same|existing|generated|query|sql)\b',
        re.IGNORECASE
    )

    def __init__(self, db_manager, schema_prefix: str = "sql_rag."):
        """
        Initialize PostgresTextMemory.

        Args:
            db_manager: DatabaseManager instance (must already be connected)
            schema_prefix: Schema prefix for table names (default: "sql_rag.")
        """
        self.db_manager = db_manager
        self.schema_prefix = schema_prefix.rstrip('.') + '.' if schema_prefix else ''
        self._sessions_table = f"{self.schema_prefix}chat_sessions"
        self._messages_table = f"{self.schema_prefix}chat_messages"
        self._tables_initialized = False
        self._ensure_tables()

    # =========================================================================
    # Table Initialization
    # =========================================================================

    def _ensure_tables(self):
        """Idempotent table creation (called on first use)."""
        if self._tables_initialized:
            return
        try:
            self.init_tables()
            self._tables_initialized = True
        except Exception as e:
            logger.warning(f"PostgresTextMemory: table init failed (will retry): {e}")

    def init_tables(self) -> bool:
        """
        Create chat_sessions and chat_messages tables if they don't exist.
        Safe to call multiple times (all statements are IF NOT EXISTS).

        Returns:
            bool: True if successful
        """
        schema = self.schema_prefix.rstrip('.')
        sessions_tbl = self._sessions_table
        messages_tbl = self._messages_table

        ddl = f"""
        CREATE SCHEMA IF NOT EXISTS {schema};

        CREATE TABLE IF NOT EXISTS {sessions_tbl} (
            session_id  TEXT        PRIMARY KEY,
            user_id     TEXT        NOT NULL,
            started_at  TIMESTAMPTZ DEFAULT NOW(),
            ended_at    TIMESTAMPTZ,
            is_active   BOOLEAN     DEFAULT TRUE,
            metadata    JSONB       DEFAULT '{{}}' ::JSONB
        );

        CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id
            ON {sessions_tbl}(user_id);

        CREATE TABLE IF NOT EXISTS {messages_tbl} (
            id                BIGSERIAL   PRIMARY KEY,
            user_id           INTEGER     NOT NULL,
            session_id        TEXT        NOT NULL
                                  REFERENCES {sessions_tbl}(session_id)
                                  ON DELETE CASCADE,
            role              TEXT        NOT NULL
                                  CHECK (role IN ('user', 'assistant', 'sql')),
            content           TEXT        NOT NULL,
            sql_generated     JSONB,
            feedback_summary  TEXT,
            search_tsv        TSVECTOR
                                  GENERATED ALWAYS AS
                                  (to_tsvector('english', content)) STORED,
            created_at        TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_chat_messages_tsv
            ON {messages_tbl} USING GIN(search_tsv);

        CREATE INDEX IF NOT EXISTS idx_chat_messages_user_created
            ON {messages_tbl}(user_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_chat_messages_user_sql
            ON {messages_tbl}(user_id, created_at DESC)
            WHERE role = 'sql';
        """

        try:
            # Execute each statement separately (some drivers need this)
            statements = [s.strip() for s in ddl.split(';') if s.strip()]
            with self.db_manager.engine.connect() as conn:
                for stmt in statements:
                    conn.execute(text(stmt))
                conn.commit()
            logger.info("PostgresTextMemory: tables initialized successfully")
            self._tables_initialized = True
            return True
        except Exception as e:
            logger.error(f"PostgresTextMemory.init_tables() failed: {e}")
            return False

    # =========================================================================
    # Session Management
    # =========================================================================

    def create_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """
        Create or register a chat session for a user.

        Args:
            user_id: Normalized user ID ("emp:123" or "mob:9876543210")
            session_id: Optional explicit session ID; auto-generated if None

        Returns:
            session_id string
        """
       # self._ensure_tables()
        if not session_id:
            session_id = f"sess_{user_id}_{uuid.uuid4().hex[:12]}"

        try:
            sql = text(f"""
                INSERT INTO {self._sessions_table} (session_id, user_id, started_at, is_active)
                VALUES (:sid, :uid, NOW(), TRUE)
                ON CONFLICT (session_id) DO NOTHING
            """)
            with self.db_manager.engine.connect() as conn:
                conn.execute(sql, {"sid": session_id, "uid": user_id})
                conn.commit()
            logger.info(f"Session created: {session_id} for user {user_id}")
        except Exception as e:
            logger.error(f"PostgresTextMemory.create_session() failed: {e}")

        return session_id

    # =========================================================================
    # Core Memory Operations
    # =========================================================================

    def save_message(
        self,
        user_id: str,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        sql_generated: Optional[Dict[str, Any]] = None,
        feedback: Optional[str] = None,
    ) -> bool:
        """
        Insert a message into chat_messages. TSVECTOR auto-generated by Postgres.

        Args:
            user_id:       Normalized user ID ("emp:123" or "mob:9876543210")
            role:          One of 'user', 'assistant', 'sql'
            content:       Text content of the message
            session_id:    Optional session ID; if None, uses/creates default session
            sql_generated: Optional dict with structured SQL info
                           {"query": "SELECT ...", "tables": [...], "modification_type": "..."}
            feedback:      Optional extracted feedback/modification tag string

        Returns:
            bool: True on success
        """
        self._ensure_tables()

        if not session_id:
            session_id = self._get_or_create_default_session(user_id)
        else:
            self.create_session(user_id, session_id)

        sql_json = json.dumps(sql_generated) if sql_generated else None
        feedback_str = feedback or self.extract_feedback(content)

        try:
            stmt = text(f"""
                INSERT INTO {self._messages_table}
                    (user_id, session_id, role, content, sql_generated, feedback_summary, created_at)
                VALUES
                    (:user_id, :session_id, :role, :content, CAST(:sql_generated AS JSONB), :feedback, NOW())
            """)
            with self.db_manager.engine.connect() as conn:
                conn.execute(stmt, {
                    "user_id": user_id,
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                    "sql_generated": sql_json,
                    "feedback": feedback_str or None,
                })
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"PostgresTextMemory.save_message() failed: {e}")
            return False

    def get_recent_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch the most recent messages for a user in chronological order.

        Args:
            user_id: Normalized user ID
            limit:   Max number of messages to retrieve

        Returns:
            List of message dicts ordered oldest-first
        """
        self._ensure_tables()
        try:
            stmt = text(f"""
                SELECT id, user_id, session_id, role, content, sql_generated,
                       feedback_summary, created_at
                FROM {self._messages_table}
                WHERE user_id = :user_id
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            with self.db_manager.engine.connect() as conn:
                result = conn.execute(stmt, {"user_id": user_id, "limit": limit})
                rows = result.fetchall()
                cols = result.keys()

            messages = [dict(zip(cols, row)) for row in rows]
            # Reverse to get chronological order (oldest first)
            messages.reverse()

            # Parse JSONB fields
            for msg in messages:
                if msg.get('sql_generated') and isinstance(msg['sql_generated'], str):
                    try:
                        msg['sql_generated'] = json.loads(msg['sql_generated'])
                    except Exception:
                        pass
                # Ensure datetime is serializable
                if hasattr(msg.get('created_at'), 'isoformat'):
                    msg['created_at'] = msg['created_at'].isoformat()

            return messages
        except Exception as e:
            logger.error(f"PostgresTextMemory.get_recent_history() failed: {e}")
            return []

    def get_relevant_context(
        self, user_id: str, current_query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: recent messages UNION FTS-ranked matches.

        Strategy:
        1. Fetch most recent `limit` messages (recency)
        2. Run full-text search for current_query keywords against all history
        3. Merge and deduplicate keeping FTS matches even if >limit turns back
        4. Return unified list, chronological order

        Args:
            user_id:       Normalized user ID
            current_query: Current user question (keywords extracted for FTS)
            limit:         Max RECENT messages to include in baseline set

        Returns:
            Combined, deduplicated list of relevant messages
        """
        self._ensure_tables()

        # Extract meaningful search terms (remove stopwords)
        search_terms = self._extract_search_terms(current_query)
        ts_query = " | ".join(search_terms) if search_terms else current_query

        try:
            recent_half = max(limit // 2, 5)
            fts_half = max(limit - recent_half, 5)

            stmt = text(f"""
                WITH recent AS (
                    SELECT id, user_id, session_id, role, content, sql_generated,
                           feedback_summary, created_at, 1.0 AS relevance_score
                    FROM {self._messages_table}
                    WHERE user_id = :user_id
                    ORDER BY created_at DESC
                    LIMIT :recent_limit
                ),
                fts_matches AS (
                    SELECT id, user_id, session_id, role, content, sql_generated,
                           feedback_summary, created_at,
                           ts_rank(search_tsv, to_tsquery('english', :ts_query)) AS relevance_score
                    FROM {self._messages_table}
                    WHERE user_id = :user_id
                      AND search_tsv @@ to_tsquery('english', :ts_query)
                    ORDER BY relevance_score DESC
                    LIMIT :fts_limit
                ),
                combined AS (
                    SELECT * FROM recent
                    UNION
                    SELECT * FROM fts_matches
                )
                SELECT DISTINCT ON (id) id, user_id, session_id, role, content,
                       sql_generated, feedback_summary, created_at, relevance_score
                FROM combined
                ORDER BY id ASC, relevance_score DESC
            """)

            with self.db_manager.engine.connect() as conn:
                result = conn.execute(stmt, {
                    "user_id": user_id,
                    "recent_limit": recent_half,
                    "ts_query": ts_query,
                    "fts_limit": fts_half,
                })
                rows = result.fetchall()
                cols = result.keys()

            messages = [dict(zip(cols, row)) for row in rows]

            # Parse JSONB and datetime
            for msg in messages:
                if msg.get('sql_generated') and isinstance(msg['sql_generated'], str):
                    try:
                        msg['sql_generated'] = json.loads(msg['sql_generated'])
                    except Exception:
                        pass
                if hasattr(msg.get('created_at'), 'isoformat'):
                    msg['created_at'] = msg['created_at'].isoformat()

            return messages

        except Exception as e:
            logger.warning(f"FTS retrieval failed ({e}), falling back to recent history")
            return self.get_recent_history(user_id, limit=limit)

    def extract_feedback(self, content: str) -> Optional[str]:
        """
        Detect modification intent in user message via keyword/regex matching.

        Checks for combinations of:
        - Action words: fix, modify, change, wrong, correct, etc.
        - Target words: join, filter, where, group by, column, etc.
        - Reference words: previous, last, prior, that query, etc.

        Args:
            content: User message text

        Returns:
            Short tag string describing detected modification (e.g. "fix:join"),
            or None if no modification intent detected
        """
        if not content:
            return None

        content_lower = content.lower()

        has_action = bool(self._ACTION_WORDS.search(content_lower))
        has_target = bool(self._TARGET_WORDS.search(content_lower))
        has_reference = bool(self._REFERENCE_WORDS.search(content_lower))

        # Strong signal: action + target OR action + reference
        if has_action and (has_target or has_reference):
            action_match = self._ACTION_WORDS.search(content_lower)
            target_match = self._TARGET_WORDS.search(content_lower)
            action = action_match.group(1) if action_match else 'modify'
            target = re.sub(r'\s+', '_', target_match.group(0).strip()) if target_match else 'query'
            return f"{action}:{target}"

        # Medium signal: reference + target (e.g., "previous join" → implicit fix)
        if has_reference and has_target:
            target_match = self._TARGET_WORDS.search(content_lower)
            target = re.sub(r'\s+', '_', target_match.group(0).strip()) if target_match else 'query'
            return f"modify:{target}"

        # Explicit modification phrases
        explicit_patterns = [
            (r'\badd\s+(a\s+)?(where|filter|condition|having)\b', 'add:filter'),
            (r'\badd\s+(a\s+)?column\b', 'add:column'),
            (r'\buse\s+\w+\s+instead\b', 'replace:field'),
            (r'\bswitch\s+(to|from)\b', 'replace:field'),
        ]
        for pattern, tag in explicit_patterns:
            if re.search(pattern, content_lower):
                return tag

        return None

    def get_last_sql(self, user_id: str) -> Optional[str]:
        """
        Retrieve the SQL query text from the most recent 'sql' role message.

        Args:
            user_id: Normalized user ID

        Returns:
            SQL query string, or None if no prior SQL found
        """
        self._ensure_tables()
        try:
            stmt = text(f"""
                SELECT sql_generated, content
                FROM {self._messages_table}
                WHERE user_id = :user_id
                  AND role = 'sql'
                ORDER BY created_at DESC
                LIMIT 1
            """)
            with self.db_manager.engine.connect() as conn:
                result = conn.execute(stmt, {"user_id": user_id})
                row = result.fetchone()

            if row:
                sql_generated = row[0]
                content = row[1]

                # Try structured JSONB first
                if sql_generated:
                    if isinstance(sql_generated, str):
                        try:
                            sql_generated = json.loads(sql_generated)
                        except Exception:
                            pass
                    if isinstance(sql_generated, dict):
                        return sql_generated.get('query') or content
                # Fallback to raw content
                return content

            return None
        except Exception as e:
            logger.error(f"PostgresTextMemory.get_last_sql() failed: {e}")
            return None

    # =========================================================================
    # Prompt Formatting
    # =========================================================================

    def format_for_prompt(self, messages: List[Dict[str, Any]], max_sql_chars: int = 800) -> str:
        """
        Render a list of messages into a readable conversation history string
        suitable for inclusion in agent prompts.

        Args:
            messages:      Output of get_recent_history() or get_relevant_context()
            max_sql_chars: Truncate SQL blocks longer than this to avoid bloating prompts

        Returns:
            Formatted multi-line string
        """
        if not messages:
            return "No previous conversation."

        lines = []
        turn_counter = 0
        pending_user = None

        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            ts = msg.get('created_at', '')
            sql_block = msg.get('sql_generated')
            feedback = msg.get('feedback_summary')

            if role == 'user':
                turn_counter += 1
                pending_user = content
                lines.append(f"\n--- Turn {turn_counter} ---")
                lines.append(f"User: {content}")

            elif role == 'assistant':
                lines.append(f"Assistant: {content}")
                if feedback:
                    lines.append(f"  [Feedback detected: {feedback}]")

            elif role == 'sql':
                # Prefer structured JSON, fallback to content
                sql_text = content
                if sql_block and isinstance(sql_block, dict):
                    sql_text = sql_block.get('query', content)
                # Truncate if too long
                if len(sql_text) > max_sql_chars:
                    sql_text = sql_text[:max_sql_chars] + "\n  ... (truncated)"
                lines.append(f"Generated SQL:\n  {sql_text}")

        return "\n".join(lines) if lines else "No previous conversation."

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _get_or_create_default_session(self, user_id: str) -> str:
        """Get the active session for user or create a new default one."""
        try:
            stmt = text(f"""
                SELECT session_id FROM {self._sessions_table}
                WHERE user_id = :user_id AND is_active = TRUE
                ORDER BY started_at DESC LIMIT 1
            """)
            with self.db_manager.engine.connect() as conn:
                result = conn.execute(stmt, {"user_id": user_id})
                row = result.fetchone()
            if row:
                return row[0]
        except Exception:
            pass
        return self.create_session(user_id)

    def _extract_search_terms(self, query: str, min_len: int = 3) -> List[str]:
        """
        Extract meaningful keywords from a query for ts_query construction.
        Removes common SQL/English stopwords and short tokens.
        """
        # PostgreSQL English stopwords + SQL keywords we don't want to FTS on
        stopwords = {
            'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to', 'for',
            'of', 'and', 'or', 'but', 'not', 'with', 'from', 'by', 'as',
            'me', 'my', 'i', 'we', 'you', 'all', 'can', 'do', 'get', 'be',
            'show', 'give', 'list', 'find', 'how', 'what', 'which'
        }
        words = re.findall(r'[a-zA-Z_]+', query.lower())
        terms = [w for w in words if len(w) >= min_len and w not in stopwords]

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        return unique[:10]  # Limit to 10 terms for ts_query safety
