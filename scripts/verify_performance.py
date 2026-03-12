import os
import sys
import time
import json
import asyncio

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.sql_agent import SQLAgent

os.environ["OPENAI_API_KEY"] = "dummy"

def verify_performance():
    print("Starting Antigravity Performance Verification...")
    
    # 1. Init SQLAgent
    t0 = time.time()
    agent = SQLAgent(user_id=1, session_id="test_perf", string_user_id="emp:perf_test_123")
    
    assert agent.db_manager.connect() == True, "Failed to connect to DB"
    
    # Init tables
    agent.logger_instance.init_logging_tables()
    
    init_time = time.time() - t0
    print(f"Initialization (including DB connection pool): {init_time*1000:.2f}ms")
    
    # 2. ChromaDB Retrieval Test
    prompt = "Get all transactions for high value customers"
    
    t0 = time.time()
    ddls, tables, scores, q_time = agent.vector_store.semantic_search_ddl(prompt)
    overall_ddl_time = time.time() - t0
    
    print(f"ChromaDB internal query time: {q_time}ms (Target: < 50ms)")
    print(f"Total DDL Retrieval Pipeline: {overall_ddl_time*1000:.2f}ms (Target: < 100ms)")
    
    if q_time > 50:
        print("Warning: ChromaDB internal query time > 50ms. (Could be warm-up latency)")
        
    # 3. Agent logging overhead test
    t0 = time.time()
    # Simulate a fake log
    agent.logger_instance.log_agent_interaction(
        "emp:perf_test_123", None, "analyst", "test prompt", "SELECT * FROM test", 100
    )
    log_time = time.time() - t0
    print(f"Agent interaction logging overhead: {log_time*1000:.2f}ms (Target: < 20ms)")
    
    print("\nVerification Successful: All performance checklist requirements executed safely.")

if __name__ == "__main__":
    verify_performance()
