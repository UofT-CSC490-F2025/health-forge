import sqlite3
import pandas as pd
from pathlib import Path
from slugify import slugify 

db_path = Path(__file__).parent / "MIMIC_IV_demo.sqlite"
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()
print("Tables:", tables)

