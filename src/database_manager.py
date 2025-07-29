import json
# from pydoc import text
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, MetaData, inspect, text
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
            
            # Create SQLAlchemy engine
            url = URL.create(
                drivername="postgresql+psycopg2",
                username=db_config['username'],
                password=db_config['password'],
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database']
            )
            print(url)

         
            self.engine = create_engine(url)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Load metadata
            self.metadata = MetaData()
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
            
            for table_name in inspector.get_table_names():
                table_info = {
                    "name": table_name,
                    "columns": [],
                    "primary_keys": [],
                    "foreign_keys": []
                }
                
                # Get columns
                for column in inspector.get_columns(table_name):
                    table_info["columns"].append({
                        "name": column['name'],
                        "type": str(column['type']),
                        "nullable": column['nullable'],
                        "default": column['default']
                    })
                
                # Get primary keys
                pk_constraint = inspector.get_pk_constraint(table_name)
                if pk_constraint['constrained_columns']:
                    table_info["primary_keys"] = pk_constraint['constrained_columns']
                
                # Get foreign keys
                fk_constraints = inspector.get_foreign_keys(table_name)
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
    
    def execute_query(self, sql_query: str) -> Optional[pd.DataFrame]:
        """Execute SQL query and return results as DataFrame."""
        if not self.engine:
            raise Exception("Database not connected. Call connect() first.")
        
        try:
            df = pd.read_sql_query(sql_query, self.engine)
            return df
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> Optional[pd.DataFrame]:
        """Get sample data from a specific table."""
        if not self.engine:
            raise Exception("Database not connected. Call connect() first.")
        
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            return pd.read_sql_query(query, self.engine)
        except Exception as e:
            logger.error(f"Error getting sample data from {table_name}: {str(e)}")
            return None
    
    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed") 