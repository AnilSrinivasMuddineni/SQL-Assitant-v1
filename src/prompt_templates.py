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

# Master prompt template with all context placeholders
MASTER_PROMPT = """
You are an expert SQL developer working with PostgreSQL databases.

## Conversation History:
{conversation_history}

## Database Schema (with column descriptions):
{schema_with_comments}

## Current User Query:
{current_query}

## Previous Feedback (if any):
{user_feedback}

## Instructions:
Generate a PostgreSQL query that addresses the current request. 
Take into account:
1. The conversation history to understand context
2. The database schema with column descriptions
3. Any feedback provided to correct previous attempts

Output ONLY the raw SQL query without any explanations, markdown formatting, or additional text.
"""

# Analysis task template
ANALYSIS_TASK_TEMPLATE = """
Analyze the following natural language query and identify:
1. The main entities/tables involved
2. The type of operation (SELECT, INSERT, UPDATE, DELETE)
3. Any filtering conditions (WHERE, HAVING)
4. Any aggregation requirements (COUNT, SUM, AVG, etc.)
5. Any sorting requirements

## Conversation Context:
{conversation_history}

## Current Query: 
{current_query}

## Database Schema Context (DDLs):
{schema_context}

Provide a detailed analysis in JSON format:
{{
    "entities": ["list of main tables"],
    "operation": "SELECT/INSERT/UPDATE/DELETE",
    "filters": ["list of filtering conditions"],
    "aggregations": ["list of aggregation functions needed"],
    "sorting": ["list of sorting requirements"],
    "context_notes": "any relevant notes from conversation history"
}}
"""

# Schema task template
SCHEMA_TASK_TEMPLATE = """
Based on the analysis, provide detailed database context including:
1. Table relationships and foreign keys based on DDLs
2. Data types and constraints
3. Indexing considerations
4. Column descriptions and meanings

## Schema Context (DDLs with comments):
{schema_context}

## Previous Feedback:
{user_feedback}

Provide database-specific insights for SQL generation, paying special attention to:
- Column names and their descriptions
- Relationships between tables
- Any corrections mentioned in feedback
"""

# Generation task template
GENERATION_TASK_TEMPLATE = """
Generate a valid PostgreSQL SQL query for the following request.

## Current User Query:
{current_query}

## Conversation History:
{conversation_history}

## Schema Context (with column descriptions):
{schema_context}

## Previous Feedback/Corrections:
{user_feedback}

CRITICAL INSTRUCTIONS:
- Output ONLY the raw SQL query
- NO "Thought:", "Final Answer:", or explanations
- NO markdown formatting (no ```sql)
- Start the output directly with the SQL verb (SELECT, INSERT, etc.)
- Do not include "I now can give a great answer"
- If feedback mentions corrections, apply them to this query
- Pay attention to column names mentioned in feedback
"""


def format_conversation_history(history: list) -> str:
    """
    Format conversation history for inclusion in prompts.
    
    Args:
        history: List of conversation turns with queries and SQL
        
    Returns:
        Formatted string of conversation history
    """
    if not history:
        return "No previous conversation."
    
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
        feedback: Optional feedback text
        
    Returns:
        Formatted feedback string
    """
    if not feedback:
        return "No feedback provided."
    
    return f"User Feedback: {feedback}\nPlease address this feedback in the current query generation."


def prepare_master_context(conversation_history: str, schema_with_comments: str, 
                          current_query: str, user_feedback: str = None) -> str:
    """
    Prepare the complete master prompt with all context filled in.
    
    Args:
        conversation_history: Formatted conversation history
        schema_with_comments: Database schema with column comments
        current_query: Current user query
        user_feedback: Optional user feedback
        
    Returns:
        Complete formatted master prompt
    """
    return MASTER_PROMPT.format(
        conversation_history=conversation_history,
        schema_with_comments=schema_with_comments,
        current_query=current_query,
        user_feedback=format_user_feedback(user_feedback)
    )
