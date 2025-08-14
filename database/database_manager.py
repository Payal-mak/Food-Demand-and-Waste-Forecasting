# database_manager.py
import sqlite3

def create_table(db_path='predictions.db'):
    """Creates the 'predictions' table if it doesn't already exist."""
    conn = sqlite3.connect(db_path)
    print("Creating 'predictions' table if it does not exist...")
    query = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        week TEXT NOT NULL,
        demand REAL NOT NULL,
        waste REAL NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        conn.execute(query)
        conn.commit()
        print("Table created successfully or already exists.")
    except Exception as e:
        print(f"An error occurred during table creation: {e}")
    finally:
        conn.close()

def save_prediction(db_path, week, demand, waste):
    """Saves a new prediction record to the database."""
    conn = sqlite3.connect(db_path)
    query = "INSERT INTO predictions (week, demand, waste) VALUES (?, ?, ?)"
    try:
        conn.execute(query, (week, demand, waste))
        conn.commit()
        print(f"Saved prediction for week {week} to the database.")
    except Exception as e:
        print(f"An error occurred during prediction save: {e}")
    finally:
        conn.close()