import json
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in configuration file: {config_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def save_config(config_path: str, config: Dict[str, Any]) -> bool:
    """
    Save configuration to a JSON file.
    
    Args:
        config_path (str): Path to the configuration file.
        config (Dict[str, Any]): Configuration dictionary to save.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving config: {str(e)}")
        return False

def format_time_taken(start_time: float) -> str:
    """
    Format the time taken for an operation.
    
    Args:
        start_time (float): Start time in seconds.
        
    Returns:
        str: Formatted string (e.g., "1.23s").
    """
    elapsed = time.time() - start_time
    minutes = elapsed / 60
    return f"{minutes:.2f}m"

def init_settings_table(db_manager) -> bool:
    """
    Initialize the user_settings table if it doesn't exist.
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS user_settings (
        setting_key VARCHAR(255) PRIMARY KEY,
        setting_value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        db_manager.execute_query(create_table_sql)
        return True
    except Exception as e:
        logger.error(f"Error creating user_settings table: {str(e)}")
        return False

def load_settings_from_db(db_manager) -> Dict[str, Any]:
    """
    Load settings from the database.
    """
    try:
        query = "SELECT setting_key, setting_value FROM user_settings;"
        df = db_manager.execute_query(query)
        settings = {}
        for _, row in df.iterrows():
            try:
                # Try to parse JSON values if possible, otherwise keep as string
                settings[row['setting_key']] = json.loads(row['setting_value'])
            except json.JSONDecodeError:
                settings[row['setting_key']] = row['setting_value']
        return settings
    except Exception as e:
        logger.error(f"Error loading settings from DB: {str(e)}")
        return {}

def save_settings_to_db(db_manager, settings: Dict[str, Any]) -> bool:
    """
    Save settings to the database (Upsert).
    """
    try:
        # Ensure table exists
        init_settings_table(db_manager)
        
        for key, value in settings.items():
            # Convert value to JSON string if it's a dict/list, otherwise string
            if isinstance(value, (dict, list)):
                val_str = json.dumps(value)
            else:
                val_str = str(value)
            
            # Upsert query using parameters
            upsert_sql = """
            INSERT INTO user_settings (setting_key, setting_value, updated_at)
            VALUES (:key, :value, CURRENT_TIMESTAMP)
            ON CONFLICT (setting_key) 
            DO UPDATE SET 
                setting_value = EXCLUDED.setting_value,
                updated_at = CURRENT_TIMESTAMP;
            """
            
            db_manager.execute_query(upsert_sql, params={"key": key, "value": val_str})
            
        return True
    except Exception as e:
        logger.error(f"Error saving settings to DB: {str(e)}")
        return False

def init_feedback_table(db_manager) -> bool:
    """
    Initialize the feedback table if it doesn't exist.
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS feedback (
        id SERIAL PRIMARY KEY,
        natural_language_query TEXT,
        generated_sql TEXT,
        rating VARCHAR(10), -- 'positive' or 'negative'
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        db_manager.execute_query(create_table_sql)
        return True
    except Exception as e:
        logger.error(f"Error creating feedback table: {str(e)}")
        return False

def save_feedback_to_db(db_manager, query: str, sql: str, rating: str) -> bool:
    """
    Save user feedback to the database.
    """
    try:
        # Ensure table exists
        init_feedback_table(db_manager)
        
        insert_sql = """
        INSERT INTO feedback (natural_language_query, generated_sql, rating)
        VALUES (:query, :sql, :rating);
        """
        
        db_manager.execute_query(insert_sql, params={
            "query": query,
            "sql": sql,
            "rating": rating
        })
        return True
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        return False

def init_cache_table(db_manager) -> bool:
    """
    Initialize the query cache table if it doesn't exist.
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS query_cache (
        query_hash VARCHAR(64) PRIMARY KEY,
        natural_language_query TEXT,
        sql_query TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        db_manager.execute_query(create_table_sql)
        return True
    except Exception as e:
        logger.error(f"Error creating query_cache table: {str(e)}")
        return False

def get_cached_query(db_manager, query_hash: str) -> Optional[str]:
    """
    Retrieve a cached SQL query by hash.
    """
    try:
        # Ensure table exists
        init_cache_table(db_manager)
        
        query = f"SELECT sql_query FROM query_cache WHERE query_hash = '{query_hash}';"
        df = db_manager.execute_query(query)
        
        if not df.empty:
            return df.iloc[0]['sql_query']
        return None
    except Exception as e:
        logger.error(f"Error retrieving cached query: {str(e)}")
        return None

def cache_query(db_manager, query_hash: str, nl_query: str, sql_query: str) -> bool:
    """
    Cache a generated SQL query.
    """
    try:
        insert_sql = """
        INSERT INTO query_cache (query_hash, natural_language_query, sql_query)
        VALUES (:hash, :nl_query, :sql_query)
        ON CONFLICT (query_hash) DO NOTHING;
        """
        
        db_manager.execute_query(insert_sql, params={
            "hash": query_hash,
            "nl_query": nl_query,
            "sql_query": sql_query
        })
        return True
    except Exception as e:
        logger.error(f"Error caching query: {str(e)}")
        return False
