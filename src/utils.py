import json
import logging
import time
import uuid
import secrets
import re
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

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
    return f"{minutes:.2f} mins"

def get_schema_prefix(db_manager) -> str:
    """
    Get the schema prefix for table names based on configuration.
    
    Args:
        db_manager: DatabaseManager instance
        
    Returns:
        str: Schema prefix (e.g., "gen_ai.") or empty string if not configured.
    """
    try:
        if hasattr(db_manager, 'config'):
            config = db_manager.config
            # Try config_management first per user request
            schema = config.get('config_management', {}).get('schema_name')
            if not schema:
                # Fallback to database config
                schema = config.get('database', {}).get('schema')
            
            if schema:
                return f"{schema}."
    except Exception:
        pass
    return ""

def init_settings_table(db_manager) -> bool:
    """
    Initialize the user_settings table if it doesn't exist.
    """
    schema_prefix = get_schema_prefix(db_manager)
    table_name = f"{schema_prefix}user_settings"
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
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
        schema_prefix = get_schema_prefix(db_manager)
        table_name = f"{schema_prefix}user_settings"
        
        query = f"SELECT setting_key, setting_value FROM {table_name};"
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
        
        schema_prefix = get_schema_prefix(db_manager)
        table_name = f"{schema_prefix}user_settings"
        
        for key, value in settings.items():
            # Convert value to JSON string if it's a dict/list, otherwise string
            if isinstance(value, (dict, list)):
                val_str = json.dumps(value)
            else:
                val_str = str(value)
            
            # Upsert query using parameters
            upsert_sql = f"""
            INSERT INTO {table_name} (setting_key, setting_value, updated_at)
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
    schema_prefix = get_schema_prefix(db_manager)
    table_name = f"{schema_prefix}feedback"
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
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
        
        schema_prefix = get_schema_prefix(db_manager)
        table_name = f"{schema_prefix}feedback"
        
        insert_sql = f"""
        INSERT INTO {table_name} (natural_language_query, generated_sql, rating)
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
    schema_prefix = get_schema_prefix(db_manager)
    table_name = f"{schema_prefix}query_cache"
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
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
        
        schema_prefix = get_schema_prefix(db_manager)
        table_name = f"{schema_prefix}query_cache"
        
        query = f"SELECT sql_query FROM {table_name} WHERE query_hash = '{query_hash}';"
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
        schema_prefix = get_schema_prefix(db_manager)
        table_name = f"{schema_prefix}query_cache"
        
        insert_sql = f"""
        INSERT INTO {table_name} (query_hash, natural_language_query, sql_query)
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


# ============================================================================
# USER ID NORMALIZATION
# ============================================================================

def normalize_user_id(emp_id: Optional[str], mobile: Optional[str]) -> Optional[str]:
    """
    Normalize inputs into a canonical user_id string.

    Priority: emp_id > mobile (if both provided, emp_id wins)
    Format:
        emp_id  -> "emp:12345"
        mobile  -> "mob:9876543210"

    Args:
        emp_id: Optional employee ID string
        mobile: Optional mobile number string

    Returns:
        Normalized user_id string, or None if neither is provided
    """
    emp_id = emp_id.strip() if emp_id else ""
    mobile = mobile.strip() if mobile else ""

    # Remove non-digit characters from mobile for normalization
    mobile_clean = re.sub(r'[^\d]', '', mobile) if mobile else ""

    if emp_id:
        return f"emp:{emp_id}"
    elif mobile_clean:
        return f"mob:{mobile_clean}"
    return None

def get_db_user_id(db_manager, emp_id: Optional[str], mobile: Optional[str]) -> Optional[int]:
    """Retrieve or create an integer user_id from user_profiles."""
    try:
        schema_prefix = get_schema_prefix(db_manager)
        profiles_table = f"{schema_prefix}user_profiles"
        emp_val = emp_id.strip() if emp_id else None
        mob_val = mobile.strip() if mobile else None

        conds = []
        params = {}
        if emp_val:
            conds.append("emp_id = :emp")
            params['emp'] = emp_val
        if mob_val:
            conds.append("mobile_number = :mob")
            params['mob'] = mob_val

        if not conds:
            return None

        where_clause = " OR ".join(conds)
        select_sql = f"SELECT user_id FROM {profiles_table} WHERE {where_clause} LIMIT 1;"
        df = db_manager.execute_query(select_sql, params=params)

        if df is not None and not df.empty:
            return int(df.iloc[0]['user_id'])

        # Insert new
        cols = []
        vals = []
        if emp_val:
            cols.append("emp_id")
            vals.append(":emp")
        if mob_val:
            cols.append("mobile_number")
            vals.append(":mob")

        insert_sql = f"INSERT INTO {profiles_table} ({', '.join(cols)}) VALUES ({', '.join(vals)}) ON CONFLICT DO NOTHING;"
        db_manager.execute_query(insert_sql, params=params)
        
        # Select again
        df_new = db_manager.execute_query(select_sql, params=params)
        if df_new is not None and not df_new.empty:
            return int(df_new.iloc[0]['user_id'])
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error in get_db_user_id: {e}")
    return None

def validate_user_input_v2(
    emp_id: Optional[str], mobile: Optional[str]
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate user input for the conversational memory flow.

    Requires either emp_id OR mobile (role is no longer sufficient).
    Returns the normalized user_id on success.

    Args:
        emp_id: Optional employee ID
        mobile: Optional mobile number

    Returns:
        Tuple of (is_valid: bool, error_message: str, user_id: Optional[str])
    """
    emp_id = emp_id.strip() if emp_id else ""
    mobile = mobile.strip() if mobile else ""
    mobile_clean = re.sub(r'[^\d]', '', mobile) if mobile else ""

    if not emp_id and not mobile_clean:
        return (
            False,
            "Please provide Employee ID or Mobile Number to identify your session.",
            None
        )

    if emp_id and len(emp_id) < 2:
        return False, "Employee ID must be at least 2 characters.", None

    if mobile_clean and len(mobile_clean) < 10:
        return False, "Mobile number must be at least 10 digits.", None

    user_id = normalize_user_id(emp_id or None, mobile_clean or None)
    return True, "", user_id


# ============================================================================
# USER AUTHENTICATION & SESSION MANAGEMENT
# ============================================================================

def init_user_tables(db_manager) -> bool:
    """
    Initialize user management tables: user_profiles, chat_sessions, and conversation_history.
    
    Args:
        db_manager: DatabaseManager instance
        
    Returns:
        bool: True if all tables created successfully, False otherwise
    """
    try:
        schema_prefix = get_schema_prefix(db_manager)
        profiles_table = f"{schema_prefix}user_profiles"
        sessions_table = f"{schema_prefix}chat_sessions"
        history_table = f"{schema_prefix}conversation_history"
        
        # Table 1: user_profiles
        create_user_profiles_sql = f"""
        CREATE TABLE IF NOT EXISTS {profiles_table} (
            user_id SERIAL PRIMARY KEY,
            emp_id VARCHAR(50) UNIQUE,
            mobile_number VARCHAR(20) UNIQUE,
            role VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            CONSTRAINT at_least_one_identifier CHECK (
                emp_id IS NOT NULL OR mobile_number IS NOT NULL OR role IS NOT NULL
            )
        );
        """
        
        # Table 2: chat_sessions
        create_chat_sessions_sql = f"""
        CREATE TABLE IF NOT EXISTS {sessions_table} (
            session_id VARCHAR(100) PRIMARY KEY,
            user_id INTEGER REFERENCES {profiles_table}(user_id) ON DELETE CASCADE,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        );
        """
        
        # Table 3: conversation_history
        create_conversation_history_sql = f"""
        CREATE TABLE IF NOT EXISTS {history_table} (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100) REFERENCES {sessions_table}(session_id) ON DELETE CASCADE,
            user_id INTEGER REFERENCES {profiles_table}(user_id) ON DELETE CASCADE,
            user_query TEXT NOT NULL,
            generated_sql TEXT,
            execution_result JSONB,
            feedback TEXT,
            feedback_type VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Execute table creation
        db_manager.execute_query(create_user_profiles_sql)
        logger.info(f"Created/verified {profiles_table} table")
        
        db_manager.execute_query(create_chat_sessions_sql)
        logger.info(f"Created/verified {sessions_table} table")
        
        db_manager.execute_query(create_conversation_history_sql)
        logger.info(f"Created/verified {history_table} table")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating user tables: {str(e)}")
        return False


def validate_user_input(emp_id: Optional[str], mobile: Optional[str], role: Optional[str]) -> Tuple[bool, str]:
    """
    Validate user input ensuring at least one field is provided and formats are correct.
    """
    # Strip whitespace
    emp_id = emp_id.strip() if emp_id else ""
    mobile = mobile.strip() if mobile else ""
    role = role.strip() if role else ""
    
    # Check if at least one field is provided
    if not emp_id and not mobile and not role:
        return False, "At least one field (Employee ID, Mobile Number, or Role) must be provided"
    
    # Validate mobile number format if provided
    if mobile:
        # Remove spaces and dashes for validation
        mobile_clean = mobile.replace(" ", "").replace("-", "")
        if not mobile_clean.isdigit() or len(mobile_clean) < 10:
            return False, "Mobile number must be at least 10 digits"
    
    # Validate emp_id format if provided (alphanumeric)
    if emp_id:
        if len(emp_id) < 2:
            return False, "Employee ID must be at least 2 characters"
    
    # Validate role if provided
    if role:
        if len(role) < 2:
            return False, "Role must be at least 2 characters"
    
    return True, ""


def register_or_get_user(db_manager, emp_id: Optional[str], 
                         mobile: Optional[str], role: Optional[str]) -> Optional[int]:
    """
    Register a new user or retrieve existing user_id.
    """
    try:
        # Ensure tables exist
        init_user_tables(db_manager)
        
        schema_prefix = get_schema_prefix(db_manager)
        profiles_table = f"{schema_prefix}user_profiles"
        
        # Clean inputs
        emp_id = emp_id.strip() if emp_id else None
        mobile = mobile.strip() if mobile else None
        role = role.strip() if role else None
        
        # Try to find existing user by emp_id or mobile
        existing_user_id = None
        
        if emp_id:
            query = f"SELECT user_id FROM {profiles_table} WHERE emp_id = :emp_id AND is_active = TRUE;"
            df = db_manager.execute_query(query, params={"emp_id": emp_id})
            if not df.empty:
                existing_user_id = int(df.iloc[0]['user_id'])
                logger.info(f"Found existing user by emp_id: {existing_user_id}")
        
        if not existing_user_id and mobile:
            query = f"SELECT user_id FROM {profiles_table} WHERE mobile_number = :mobile AND is_active = TRUE;"
            df = db_manager.execute_query(query, params={"mobile": mobile})
            if not df.empty:
                existing_user_id = int(df.iloc[0]['user_id'])
                logger.info(f"Found existing user by mobile: {existing_user_id}")
        
        # If user exists, update their info and return user_id
        if existing_user_id:
            update_sql = f"""
            UPDATE {profiles_table} 
            SET emp_id = COALESCE(:emp_id, emp_id),
                mobile_number = COALESCE(:mobile, mobile_number),
                role = COALESCE(:role, role),
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = :user_id;
            """
            db_manager.execute_query(update_sql, params={
                "emp_id": emp_id,
                "mobile": mobile,
                "role": role,
                "user_id": existing_user_id
            })
            return existing_user_id
        
        # Create new user
        # Note: We need to handle RETURNING differently or query back
        insert_sql = f"""
        INSERT INTO {profiles_table} (emp_id, mobile_number, role)
        VALUES (:emp_id, :mobile, :role);
        """
        
        db_manager.execute_query(insert_sql, params={
            "emp_id": emp_id,
            "mobile": mobile,
            "role": role
        })
        
        # Retrieve the newly created user_id
        if emp_id:
            query = f"SELECT user_id FROM {profiles_table} WHERE emp_id = :emp_id ORDER BY created_at DESC LIMIT 1;"
            df = db_manager.execute_query(query, params={"emp_id": emp_id})
        elif mobile:
            query = f"SELECT user_id FROM {profiles_table} WHERE mobile_number = :mobile ORDER BY created_at DESC LIMIT 1;"
            df = db_manager.execute_query(query, params={"mobile": mobile})
        else:
            query = f"SELECT user_id FROM {profiles_table} WHERE role = :role ORDER BY created_at DESC LIMIT 1;"
            df = db_manager.execute_query(query, params={"role": role})
        
        if not df.empty:
            user_id = int(df.iloc[0]['user_id'])
            logger.info(f"Created new user with user_id: {user_id}")
            return user_id
        
        return None
        
    except Exception as e:
        logger.error(f"Error in register_or_get_user: {str(e)}")
        return None


def create_chat_session(db_manager, user_id: int) -> Optional[str]:
    """
    Create a new chat session for a user.
    """
    try:
        schema_prefix = get_schema_prefix(db_manager)
        sessions_table = f"{schema_prefix}chat_sessions"
        
        # Generate unique session ID
        session_id = f"session_{user_id}_{secrets.token_hex(6)}"
        
        insert_sql = f"""
        INSERT INTO {sessions_table} (session_id, user_id, started_at, is_active)
        VALUES (:session_id, :user_id, CURRENT_TIMESTAMP, TRUE);
        """
        
        db_manager.execute_query(insert_sql, params={
            "session_id": session_id,
            "user_id": user_id
        })
        
        logger.info(f"Created chat session: {session_id} for user: {user_id}")
        return session_id
        
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        return None


def generate_session_id() -> str:
    """
    Generate a unique session ID.
    """
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
