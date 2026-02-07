"""Example: compressing source code with semcompress.

NOTE: First run will download the all-MiniLM-L6-v2 model (~80MB).
"""

from semcompress import compact

# Sample Python module
source_code = '''
import os
import sys
from pathlib import Path

def load_config(path: str) -> dict:
    """Load configuration from a JSON file."""
    import json
    with open(path) as f:
        return json.load(f)

def validate_config(config: dict) -> bool:
    """Validate that all required keys are present."""
    required = ["host", "port", "database"]
    return all(k in config for k in required)

def format_connection_string(config: dict) -> str:
    """Build a database connection string from config."""
    return f"{config['host']}:{config['port']}/{config['database']}"

class DatabaseClient:
    """Client for connecting to the database."""

    def __init__(self, config: dict):
        self.config = config
        self.connection = None

    def connect(self):
        """Establish connection to the database."""
        conn_str = format_connection_string(self.config)
        self.connection = self._create_connection(conn_str)
        return self

    def _create_connection(self, conn_str: str):
        """Internal: create the raw connection object."""
        return {"url": conn_str, "status": "connected"}

    def query(self, sql: str) -> list:
        """Execute a SQL query and return results."""
        if not self.connection:
            raise RuntimeError("Not connected")
        return [{"result": sql}]

    def close(self):
        """Close the database connection."""
        self.connection = None

    def __repr__(self):
        status = "connected" if self.connection else "disconnected"
        return f"DatabaseClient({status})"

def setup_logging(level: str = "INFO"):
    """Configure logging for the application."""
    import logging
    logging.basicConfig(level=getattr(logging, level))

def main():
    """Entry point for the application."""
    config = load_config("config.json")
    if not validate_config(config):
        print("Invalid config")
        return
    client = DatabaseClient(config)
    client.connect()
    results = client.query("SELECT * FROM users")
    print(results)
    client.close()
'''

# Compress to 50% — auto-detects code mode
result = compact(source_code, ratio=0.5)

print("=" * 60)
print("CODE COMPACTION RESULT")
print("=" * 60)
print(f"Original:   {result.original_tokens} tokens")
print(f"Compressed: {result.compacted_tokens} tokens ({result.ratio:.0%})")
print(f"Removed:    {result.chunks_removed} blocks")
print(f"Kept:       {result.chunks_kept} blocks")
print(f"Iterations: {result.iterations}")
print("=" * 60)
print()
print(result.text)
