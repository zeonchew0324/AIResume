from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

DB_URL = os.getenv("DB_URL", "").replace("+asyncpg", "+psycopg2")

engine = create_engine(DB_URL, connect_args={"gssencmode": "disable"})

try:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
        print("Connection successful!")
except Exception as e:
    print(f"Failed to connect: {e}")
