import aiosqlite
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'logs.db')

async def init_db() : #creates table if it doesnt exists yet
    async with aiosqlite.connect(DB_PATH) as db :
        await db.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT,
                predicted_label TEXT,
                confidence REAL,
                uncertain BOOLEAN,
                timestamp TEXT
            )
        """)
        await db.commit()

async def log_request(input_text, predicted_label, confidence, uncertain) :
    async with aiosqlite.connect(DB_PATH) as db :
        timestamp = datetime.now().isoformat()
        await db.execute("""
            INSERT INTO logs(input_text, predicted_label, confidence, uncertain, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """,
            (input_text, predicted_label, confidence, uncertain, timestamp)             
        )
        await db.commit()

# to be wired in 2 places
# 1. main.py — call init_db() at startup inside the lifespan function, after the model loads
# 2. routers/predict.py — use FastAPI's BackgroundTasks to call log_request() after returning the response.