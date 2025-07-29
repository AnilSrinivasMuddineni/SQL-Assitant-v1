#!/usr/bin/env python3
"""
Test script for CrewAI SQL Agent
"""

import sys
import os
import json
from sqlalchemy import create_engine, text

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sql_agent import SQLAgent



def test_connections():
    """Test database and Ollama connections."""
    print("🔍 Testing connections...")
    
    agent = SQLAgent()
    
    # Test database connection
    print("Testing database connection...")

    
    # engine = create_engine('postgresql://postgres:root@localhost:5432/mydb')
    # with engine.connect() as conn:
    #     result = conn.execute(text('SELECT 1'))
    #     print(result.fetchone())

    db_connected = agent.connect_database()
    if db_connected:
        print("✅ Database connection successful")
    else:
        print("❌ Database connection failed")
        return False
    
    # Test Ollama connection
    print("Testing Ollama connection...")
    ollama_connected = agent.test_ollama_connection()
    if ollama_connected:
        print("✅ Ollama connection successful")
    else:
        print("❌ Ollama connection failed")
        return False
    
    return True

def test_sql_generation():
    """Test SQL generation with sample queries."""
    print("\n🚀 Testing SQL generation...")
    
    agent = SQLAgent()
    
    if not agent.connect_database():
        print("❌ Cannot test SQL generation - database not connected")
        return False
    
    # Sample queries to test
    test_queries = [
        "Show me all users",
        "Count total orders by status",
        "Find products with price greater than 100",
        "Get user details with their order count"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {query} ---")
        
        try:
            result = agent.generate_sql(query)
            
            if result["success"]:
                print(f"✅ SQL generated successfully")
                print(f"Generated SQL: {result['sql_query']}")
                
                # Test execution (optional)
                if input("Execute this query? (y/n): ").lower() == 'y':
                    exec_result = agent.execute_sql(result['sql_query'])
                    if exec_result["success"]:
                        print(f"✅ Query executed successfully - {exec_result['row_count']} rows returned")
                    else:
                        print(f"❌ Query execution failed: {exec_result.get('error')}")
            else:
                print(f"❌ SQL generation failed: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ Error during test: {str(e)}")
    
    return True

def test_schema_extraction():
    """Test database schema extraction."""
    print("\n📊 Testing schema extraction...")
    
    agent = SQLAgent()
    
    if not agent.connect_database():
        print("❌ Cannot test schema extraction - database not connected")
        return False
    
    try:
        schema = agent.db_manager.get_database_schema()
        print(f"✅ Schema extracted successfully")
        print(f"Total tables: {schema['total_tables']}")
        
        for table_name, table_info in schema['tables'].items():
            print(f"  - {table_name}: {len(table_info['columns'])} columns")
            
        return True
        
    except Exception as e:
        print(f"❌ Schema extraction failed: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🧪 CrewAI SQL Agent Test Suite")
    print("=" * 50)
    
    # Test connections
    if not test_connections():
        print("\n❌ Connection tests failed. Please check your setup.")
        return
    
    # Test schema extraction
    if not test_schema_extraction():
        print("\n❌ Schema extraction failed.")
        return
    
    # Test SQL generation
    if not test_sql_generation():
        print("\n❌ SQL generation tests failed.")
        return
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
    print("\n🎉 Your CrewAI SQL Agent is ready to use!")
    print("Run 'streamlit run app.py' to start the web interface.")

    sql_agent = SQLAgent()
    sql_agent._create_agents()
    print(sql_agent.agents)

if __name__ == "__main__":
    main() 