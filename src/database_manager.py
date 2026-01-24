import json
# from pydoc import text
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, MetaData, inspect, text
from sqlalchemy.schema import CreateTable
from sqlalchemy.engine import URL
import pandas as pd
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, config_path: str = "config/database_config.json"):
        """Initialize database manager with configuration."""
        self.config = self._load_config(config_path)
        self.connection = None
        self.engine = None
        self.metadata = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load database configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file: {config_path}")
            raise
    
    def connect(self) -> bool:
        """Establish connection to PostgreSQL database."""
        try:
            db_config = self.config['database']
            
            # Prepare query options for schema
            query_options = {}
            if 'schema' in db_config and db_config['schema']:
                query_options = {"options": f"-c search_path={db_config['schema']}"}

            # Create SQLAlchemy engine
            url = URL.create(
                drivername="postgresql+psycopg2",
                username=db_config['username'],
                password=db_config['password'],
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                query=query_options
            )
            print(url)

            self.engine = create_engine(url)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Load metadata
            self.metadata = MetaData()
            
            # Pass schema explicitly if defined
            if 'schema' in db_config and db_config['schema']:
                self.metadata.reflect(bind=self.engine, schema=db_config['schema'])
            else:
                self.metadata.reflect(bind=self.engine)
            
            logger.info("Database connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            return False
    
    def get_database_schema(self) -> Dict[str, Any]:
        """Extract database schema information."""
        if not self.engine:
            raise Exception("Database not connected. Call connect() first.")
        
        schema_info = {
            "tables": {},
            "relationships": [],
            "total_tables": 0
        }
        
        try:
            inspector = inspect(self.engine)
            db_config = self.config['database']
            target_schema = db_config.get('schema')
            
            # Get table names for the specific schema if defined
            if target_schema:
                table_names = inspector.get_table_names(schema=target_schema)
            else:
                table_names = inspector.get_table_names()
            
            for table_name in table_names:
                table_info = {
                    "name": table_name,
                    "columns": [],
                    "primary_keys": [],
                    "foreign_keys": []
                }
                
                # Get columns
                for column in inspector.get_columns(table_name, schema=target_schema):
                    table_info["columns"].append({
                        "name": column['name'],
                        "type": str(column['type']),
                        "nullable": column['nullable'],
                        "default": column['default']
                    })
                
                # Get primary keys
                pk_constraint = inspector.get_pk_constraint(table_name, schema=target_schema)
                if pk_constraint['constrained_columns']:
                    table_info["primary_keys"] = pk_constraint['constrained_columns']
                
                # Get foreign keys
                fk_constraints = inspector.get_foreign_keys(table_name, schema=target_schema)
                for fk in fk_constraints:
                    table_info["foreign_keys"].append({
                        "constrained_columns": fk['constrained_columns'],
                        "referred_table": fk['referred_table'],
                        "referred_columns": fk['referred_columns']
                    })
                
                schema_info["tables"][table_name] = table_info
            
            schema_info["total_tables"] = len(schema_info["tables"])
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Error extracting schema: {str(e)}")
            raise
    
    def get_relevant_tables(self, query: str) -> List[str]:
        """Extract relevant table names from a natural language query."""
        if not self.engine:
            raise Exception("Database not connected. Call connect() first.")
        
        inspector = inspect(self.engine)
        db_config = self.config['database']
        target_schema = db_config.get('schema')
        
        if target_schema:
            all_tables = inspector.get_table_names(schema=target_schema)
        else:
            all_tables = inspector.get_table_names()
        
        # Simple keyword-based table matching
        query_lower = query.lower()
        relevant_tables = []
        
        for table in all_tables:
            if table.lower() in query_lower:
                relevant_tables.append(table)
        
        # If no direct matches, return all tables for broader context
        if not relevant_tables:
            relevant_tables = all_tables[:5]  # Limit to first 5 tables
        
        return relevant_tables
    
    def execute_query(self, sql_query: str, params: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        """Execute SQL query and return results as DataFrame."""
        if not self.engine:
            raise Exception("Database not connected. Call connect() first.")
        
        try:
            # Check if it's a SELECT query
            if sql_query.strip().upper().startswith("SELECT"):
                # Use connection context and text() wrapper for consistent parameter handling
                with self.engine.connect() as conn:
                    df = pd.read_sql_query(text(sql_query), conn, params=params)
                return df
            else:
                # For non-SELECT queries (INSERT, UPDATE, CREATE, etc.)
                with self.engine.connect() as conn:
                    conn.execute(text(sql_query), params or {})
                    conn.commit()
                return None
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> Optional[pd.DataFrame]:
        """Get sample data from a specific table."""
        if not self.engine:
            raise Exception("Database not connected. Call connect() first.")
        
        try:
            # Note: We rely on search_path being set in connect() so we don't need to qualify table name here
            # unless we want to be extra safe.
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            return pd.read_sql_query(query, self.engine)
        except Exception as e:
            logger.error(f"Error getting sample data from {table_name}: {str(e)}")
            return None

    def get_all_ddls(self) -> Dict[str, str]:
        """
        Retrieve CREATE TABLE statements for all tables in the database.
        Returns a dictionary mapping table names to their DDL strings.
        """
        if not self.engine:
            raise Exception("Database not connected. Call connect() first.")

        ddls = {}
        try:
            
            # Reflect all tables
            db_config = self.config['database']
            target_schema = db_config.get('schema')
            
            if target_schema:
                self.metadata.reflect(bind=self.engine, schema=target_schema)
            else:
                self.metadata.reflect(bind=self.engine)
            
            for table_name, table in self.metadata.tables.items():
                # Generate CREATE TABLE statement
                ddl = str(CreateTable(table).compile(self.engine))
                ddls[table_name] = ddl.strip()
                
            return ddls
        except Exception as e:
            logger.error(f"Error retrieving DDLs: {str(e)}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()

    def get_configured_ddls(self, config_table_name: Optional[str] = None, table_name_column: Optional[str] = None) -> Dict[str, str]:
        """
        Retrieve DDLs ONLY for tables specified in a configuration table.
        
        Args:
            config_table_name: Optional. Overrides 'config_management.table_name' from config.
            table_name_column: Optional. Overrides 'config_management.column_name' from config.
        """
        if not self.engine:
            raise Exception("Database not connected. Call connect() first.")

        # Resolve defaults from config if not provided
        if not config_table_name:
            config_table_name = self.config.get('config_management', {}).get('table_name', 'database.config')
        
        if not table_name_column:
            table_name_column = self.config.get('config_management', {}).get('column_name', 'table_name')

        ddls = {}
        try:
            logger.info(f"Loading configured tables from {config_table_name}...")
            
            # 1. Fetch the list of tables to index
            # We use a direct SQL query to get the table names
            query = text(f"SELECT {table_name_column} FROM {config_table_name}")
            with self.engine.connect() as conn:
                try:
                    result = conn.execute(query)
                    configured_tables = [row[0] for row in result]
                except Exception as e:
                    logger.error(f"Error reading config table {config_table_name}: {e}")
                    # If config table doesn't exist, we might want to return empty or raise
                    # For now, let's log and re-raise to alert the caller
                    raise e
            
            if not configured_tables:
                logger.warning(f"No tables found in configuration {config_table_name}.")
                return {}

            logger.info(f"Found {len(configured_tables)} tables to index: {configured_tables}")

            # 2. Fetch DDL for each configured table
            for full_table_name in configured_tables:
                try:
                    # Handle schema-qualified names (e.g., "schema.table")
                    if "." in full_table_name:
                        schema_name, table_name = full_table_name.split(".", 1)
                    else:
                        schema_name = None # Use default schema or search path
                        table_name = full_table_name

                    # Use inspect to check if table exists before trying to reflect
                    inspector = inspect(self.engine)
                    # Note: has_table check might need schema
                    if inspector.has_table(table_name, schema=schema_name):
                         # Create a minimal metadata object just for this table
                        meta = MetaData()
                        # Load table definition
                        table = pd.io.sql.SQLDatabase(self.engine).get_table(table_name, schema=schema_name)
                        # We can also use sqlalchemy Table reflection directly:
                        # table = Table(table_name, meta, autoload_with=self.engine, schema=schema_name)
                        
                        # Use the existing mechanism if possible, or simple CreateTable
                        # Since we didn't import Table, let's stick to what we have or import it.
                        # Actually, we can use the main metadata but we only reflect one table at a time
                        # to avoid loading everything.
                        
                        # Let's use a fresh metadata for isolation or just reflect specific table
                        # Using a generic approach with CreateTable:
                        
                        # Reflection
                        from sqlalchemy import Table
                        t = Table(table_name, self.metadata, autoload_with=self.engine, schema=schema_name)
                        
                        ddl = str(CreateTable(t).compile(self.engine))
                        ddls[full_table_name] = ddl.strip()
                    else:
                        logger.warning(f"Configured table {full_table_name} not found in database.")

                except Exception as table_e:
                    logger.error(f"Error retrieving DDL for {full_table_name}: {table_e}")
                    continue

            return ddls

        except Exception as e:
            logger.error(f"Error in get_configured_ddls: {str(e)}")
            raise e