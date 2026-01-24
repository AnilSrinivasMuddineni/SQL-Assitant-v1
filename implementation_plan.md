# Implementation Plan - Phase 3: Advanced Features

## Goal
Enhance performance and utility by implementing query caching and data visualization.

## Proposed Changes

### 1. Query Caching
-   **Goal**: Reduce latency and LLM usage for repeated queries.
-   **Database**: Create `query_cache` table (query_hash, natural_query, sql_query, created_at).
-   **Logic**:
    -   In `SQLAgent.generate_sql`, compute hash of `natural_language_query`.
    -   Check DB for existing hash.
    -   If found -> Return cached SQL.
    -   If not -> Generate -> Save to DB -> Return.

### 2. Chart Generation
-   **Goal**: Visualize query results.
-   **UI**: Add a "Visualize" tab or section below results.
-   **Logic**:
    -   Analyze result DataFrame.
    -   If suitable (e.g., 1 numeric + 1 categorical column), suggest/render a chart.
    -   Support Bar, Line, and Pie charts using Streamlit/Plotly.

## Verification Plan
1.  **Caching**: Run same query twice. Second time should be near-instant and show "From Cache" indicator.
2.  **Charts**: Run a query like "Sales by Category" and verify a Bar Chart appears.
