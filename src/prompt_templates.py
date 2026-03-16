"""
Prompt templates for SQL Agent with conversational memory support.
"""

# Conversational System Prompt
CONVERSATIONAL_SYSTEM_PROMPT = """
You are a Conversational PostgreSQL SQL Assistant.

CORE BEHAVIOR:
1. You maintain context across multiple turns in a conversation
2. When users reference "previous query", "above SQL", "that join", "last result", etc., 
   they are asking you to MODIFY the most recent SQL you generated
3. NEVER ignore user feedback - always update SQL to reflect requested changes
4. For modifications, start with the last SQL and apply only the requested changes

MODIFICATION KEYWORDS TO WATCH FOR:
- "add [column/filter/join]"
- "remove [column/condition]"
- "fix [issue]"
- "change [aspect]"
- "use [alternative] instead"
- "also include"
- "filter by"
- "previous query"
- "last query"
- "that SQL"

OUTPUT FORMAT:
- Return ONLY valid PostgreSQL SQL
- NO explanations unless explicitly asked
- NO markdown code blocks
- Start directly with SELECT/INSERT/UPDATE/DELETE
"""

# Query Type Classifier Prompt
QUERY_TYPE_CLASSIFIER = """
Classify the following user message as either:
- NEW_QUERY: A brand new question requiring fresh SQL generation
- MODIFICATION: A request to modify/fix/enhance the previous SQL

User message: {user_message}

Last SQL generated: {last_sql}

Rules:
- If user mentions "previous", "last", "above", "that query", it's MODIFICATION
- If user says "add", "remove", "fix", "change" without context, it's likely MODIFICATION
- If it's a completely new question, it's NEW_QUERY
- If there's no last SQL, it must be NEW_QUERY

Answer with ONLY one word: NEW_QUERY or MODIFICATION
"""

# =============================================================================
# MASTER PROMPT — Used by all three CrewAI agents
# =============================================================================
MASTER_PROMPT = """
You are an expert PostgreSQL SQL developer assisting a data analyst in an iterative workflow.

{debug_info}

## Conversation History (most recent turns):
{conversation_history}

## Database Schema with Column Comments:
{schema_with_comments}

## Current User Request:
{current_query}

## Modification Instructions (if any):
{user_feedback}

## Critical Instructions:
1. **Referencing Prior SQL**: If the conversation history contains SQL queries and the current
   request is a modification (fix JOIN, add filter, change GROUP BY, etc.), you MUST start from
   the most recent SQL shown above and apply ONLY the requested change. Keep all other parts
   of the query unchanged.

2. **Schema Accuracy**: Use the column names and JOIN keys exactly as specified in the schema
   comments above. Pay special attention to lines marked with "--" comments which describe
   the business meaning and correct identifiers.

3. **Explain Changes** (when modifying): Briefly note what specific change was made compared
   to the previous SQL, then output the full modified SQL.

4. **Output Format**:
   - Output ONLY the raw SQL query (no markdown, no ```sql blocks)
   - Start directly with SELECT / INSERT / UPDATE / DELETE / WITH
   - Do NOT include "Thought:", "Final Answer:", or any preamble

5. **New Query**: If no prior SQL exists or this is a completely new question, generate fresh
   SQL based solely on the schema and current request.
"""

# Analysis task template
ANALYSIS_TASK_TEMPLATE = """
Analyze the following natural language query and identify:
1. The main entities/tables involved
2. The type of operation (SELECT, INSERT, UPDATE, DELETE)
3. Any filtering conditions (WHERE, HAVING)
4. Any aggregation requirements (COUNT, SUM, AVG, etc.)
5. Any sorting requirements
6. Whether this is a modification of a previous query

## Conversation Context:
{conversation_history}

## Current Query: 
{current_query}

## Database Schema Context (DDLs with column comments):
{schema_context}

## Modification Instructions:
{user_feedback}

Provide a detailed analysis in JSON format:
{{
    "entities": ["list of main tables"],
    "operation": "SELECT/INSERT/UPDATE/DELETE",
    "filters": ["list of filtering conditions"],
    "aggregations": ["list of aggregation functions needed"],
    "sorting": ["list of sorting requirements"],
    "is_modification": true_or_false,
    "context_notes": "any relevant notes from conversation history, especially last SQL"
}}
"""

# Schema task template
SCHEMA_TASK_TEMPLATE = """
Based on the analysis, provide detailed database context including:
1. Table relationships and foreign keys based on DDLs
2. Data types and constraints
3. Indexing considerations
4. Column descriptions and meanings

## Schema Context (DDLs with column comments):
{schema_context}

## Modification Instructions:
{user_feedback}

Provide database-specific insights for SQL generation, paying special attention to:
- Column names and their descriptions (look for -- inline comments)
- Correct JOIN keys as identified in schema comments
- Relationships between tables
- Any corrections mentioned in modification instructions
"""

# Generation task template (new queries)
GENERATION_TASK_TEMPLATE = """
Generate a valid PostgreSQL SQL query for the following request.

## Current User Query:
{current_query}

## Conversation History:
{conversation_history}

## Schema Context (with column descriptions):
{schema_context}

## Modification Instructions:
{user_feedback}

CRITICAL INSTRUCTIONS:
- Output ONLY the raw SQL query
- NO "Thought:", "Final Answer:", or explanations
- NO markdown formatting (no ```sql)
- Start the output directly with the SQL verb (SELECT, INSERT, etc.)
- Do not include "I now can give a great answer"
- If modification instructions mention corrections, apply them
- Pay attention to column names in the schema comments
"""

# Modification task template (editing prior SQL)
MODIFICATION_TASK_TEMPLATE = """
MODIFICATION REQUEST: The user wants to modify the previous SQL query.

## Previous SQL to Modify:
{last_sql}

## User's Modification Request:
{current_query}

## Conversation History (for full context):
{conversation_history}

## Schema Context (with column descriptions):
{schema_context}

## Specific Modification Instructions:
{user_feedback}

CRITICAL INSTRUCTIONS FOR MODIFICATION:
- START with the SQL shown above in "Previous SQL to Modify"
- Apply ONLY the specific changes requested by the user
- Keep ALL other parts of the query unchanged
- If user says "fix join", correct ONLY the JOIN condition
- If user says "add WHERE amount > 50000", add ONLY that filter
- If user says "change GROUP BY", update ONLY the GROUP BY clause
- Output ONLY the modified raw SQL query (no explanations, no markdown)
- Start directly with SELECT/INSERT/UPDATE/DELETE/WITH
"""


# =============================================================================
# SINGLE-AGENT MASTER PROMPT — used when SQL_AGENT_MODE=single
# =============================================================================
MASTER_SINGLE_AGENT_PROMPT = """
You are a Conversational PostgreSQL SQL Assistant.

Context:
- Conversation history:
{conversation_history}

- Enhanced database schema (with DDL + column comments):
{schema_with_comments}

- Last SQL generated for this user (if any):
{last_sql}

- Current user request:
{current_query}

- Detected user feedback (if any):
{user_feedback}

Your job (in ONE response):
1. Understand the user's intent and how it relates to the previous SQL (if provided).
2. Choose the correct tables, columns, joins, and filters using the schema and column comments.
3. Generate a single, complete PostgreSQL SQL query that satisfies the request.
4. If the user is asking to modify or correct a previous query, update that SQL instead of creating a new one.

Rules:
- Use ONLY tables and columns that exist in the provided schema context.
- Respect business rules and column meanings as described in inline -- comments.
- When fixing issues (joins, columns, filters, GROUP BY), incorporate user feedback explicitly.
- If "Last SQL" is provided and the request is a modification, START from that SQL and apply ONLY the requested changes.
- Return ONLY executable PostgreSQL SQL. Do not include explanations, comments, or natural language.
- NO markdown formatting (no \"```sql\"). Start directly with SELECT/INSERT/UPDATE/DELETE/WITH.
- Do NOT include \"Thought:\", \"Final Answer:\", or any preamble.
"""


# =============================================================================
# Helper Functions
# =============================================================================
from typing import List, Dict, Any

def format_messages_for_prompt(messages: List[Dict[str, Any]], max_sql_chars: int = 600) -> str:
    """
    Format a list of message dicts (from PostgresTextMemory) into prompt-ready string.

    Args:
        messages:      List of dicts with keys: role, content, sql_generated, feedback_summary
        max_sql_chars: Truncate SQL blocks longer than this value

    Returns:
        Formatted multi-line string for inclusion in prompts
    """
    if not messages:
        return "No previous conversation."

    lines = []

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        sql_block = msg.get('sql_generated')
        feedback = msg.get('feedback_summary')

        if role == 'user':
            lines.append(f"\nUser: {content}")
        elif role == 'assistant':
            lines.append(f"Assistant: {content}")
            if feedback:
                lines.append(f"    >> Feedback tag: {feedback}")
        elif role == 'sql':
            sql_text = str(content)
            if sql_block and isinstance(sql_block, dict):
                sql_text = str(sql_block.get('query', content))
            if len(sql_text) > max_sql_chars:
                truncated = ""
                for i, char in enumerate(sql_text):
                    if i >= max_sql_chars:
                        break
                    truncated += char
                sql_text = truncated + "\n    ... (truncated)"
            lines.append(f"    SQL:\n    {sql_text}")

    return "\n".join(lines) if lines else "No previous conversation."


def format_conversation_history(history: list) -> str:
    """
    Format conversation history for inclusion in prompts.
    Accepts both old MemoryManager format and new PostgresTextMemory format.

    Args:
        history: List of conversation turns

    Returns:
        Formatted string of conversation history
    """
    if not history:
        return "No previous conversation."

    # Detect format: new format has 'role' key, old has 'user_query'
    if history and 'role' in history[0]:
        return format_messages_for_prompt(history)

    # Legacy MemoryManager format
    formatted = []
    for i, turn in enumerate(history, 1):
        formatted.append(f"Turn {i}:")
        formatted.append(f"  User: {turn.get('user_query', 'N/A')}")
        formatted.append(f"  Generated SQL: {turn.get('generated_sql', 'N/A')}")
        if turn.get('feedback'):
            formatted.append(f"  Feedback: {turn.get('feedback')}")
        formatted.append("---")

    return "\n".join(formatted)


def format_user_feedback(feedback: str = None) -> str:
    """
    Format user feedback for inclusion in prompts.

    Args:
        feedback: Optional feedback text or modification tag

    Returns:
        Formatted feedback string
    """
    if not feedback:
        return "No modification instructions. Generate SQL from scratch if no prior SQL exists."

    return (
        f"Modification detected: {feedback}\n"
        f"Please apply this modification to the previous SQL. "
        f"Address this feedback when generating or modifying the query."
    )


def prepare_master_context(conversation_history: str, schema_with_comments: str,
                           current_query: str, user_feedback: str = None) -> str:
    """
    Prepare the complete master prompt with all context filled in.

    Args:
        conversation_history: Formatted conversation history string
        schema_with_comments: Database schema with column comments
        current_query: Current user query
        user_feedback: Optional user feedback or modification tag

    Returns:
        Complete formatted master prompt
    """
    return MASTER_PROMPT.format(
        conversation_history=conversation_history,
        schema_with_comments=schema_with_comments,
        current_query=current_query,
        user_feedback=format_user_feedback(user_feedback)
    )
