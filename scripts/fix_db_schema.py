import json
from sqlalchemy import create_engine, text
import os

config_path = 'config/database_config.json'
with open(config_path) as f:
    cfg = json.load(f)['database']

db_url = f"postgresql://{cfg['username']}:{cfg['password']}@{cfg['host']}:{cfg.get('port', 5432)}/{cfg['database']}"
engine = create_engine(db_url)

commands = [
    "DROP TABLE IF EXISTS sql_rag.chat_messages CASCADE;",
    "DROP TABLE IF EXISTS sql_rag.chat_sessions CASCADE;",
    "DROP TABLE IF EXISTS sql_rag.agent_request_response CASCADE;",
    "DROP TABLE IF EXISTS sql_rag.ddl_retrieval_log CASCADE;"
]

with engine.begin() as conn:
    for cmd in commands:
        try:
            conn.exec_driver_sql(cmd)
            print(f"Success: {cmd}")
        except Exception as e:
            print(f"Skipped {cmd}: {e}")
