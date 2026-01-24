import logging
import os
from src.database_manager import DatabaseManager
from src.utils import save_settings_to_db, save_feedback_to_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_parameterized_queries():
    print("Testing parameterized queries...")
    
    config_path = "config/database_config.json"
    db_manager = DatabaseManager(config_path)
    
    if not db_manager.connect():
        print("Failed to connect to database.")
        return

    try:
        # 1. Test execute_query with params (SELECT)
        print("1. Testing SELECT with params...")
        df = db_manager.execute_query(
            "SELECT 1 as val WHERE 1 = :val", 
            params={"val": 1}
        )
        if df is not None and not df.empty and df.iloc[0]['val'] == 1:
            print("   SUCCESS: SELECT with params worked.")
        else:
            print("   FAILURE: SELECT with params failed.")

        # 2. Test save_settings_to_db (INSERT/UPDATE with params)
        print("2. Testing save_settings_to_db (SQL Injection Safe)...")
        # Test with a value that would cause SQL injection if not parameterized
        dangerous_value = "value'); DROP TABLE test_table; --"
        settings = {
            "test_safe_key": dangerous_value
        }
        
        if save_settings_to_db(db_manager, settings):
            # Verify it was saved correctly
            saved_settings = db_manager.execute_query(
                "SELECT setting_value FROM user_settings WHERE setting_key = :key",
                params={"key": "test_safe_key"}
            )
            if not saved_settings.empty:
                val = saved_settings.iloc[0]['setting_value']
                # The value should be exactly what we put in, including the SQL injection attempt
                # It should NOT be escaped with double quotes if parameterized correctly, 
                # but it might be quoted if it was JSON dumped.
                # In utils.py: val_str = str(value) -> "value'); DROP TABLE test_table; --"
                if val == dangerous_value or val == f'"{dangerous_value}"': 
                    print(f"   SUCCESS: Value saved correctly: {val}")
                else:
                    print(f"   FAILURE: Value mismatch. Expected {dangerous_value}, got {val}")
            else:
                print("   FAILURE: Could not retrieve saved setting.")
        else:
            print("   FAILURE: save_settings_to_db returned False.")

        # 3. Test save_feedback_to_db
        print("3. Testing save_feedback_to_db...")
        if save_feedback_to_db(db_manager, "test query", "SELECT * FROM test", "positive"):
            print("   SUCCESS: save_feedback_to_db worked.")
        else:
            print("   FAILURE: save_feedback_to_db returned False.")

    except Exception as e:
        print(f"An error occurred during testing: {e}")
    finally:
        db_manager.close()

if __name__ == "__main__":
    test_parameterized_queries()
