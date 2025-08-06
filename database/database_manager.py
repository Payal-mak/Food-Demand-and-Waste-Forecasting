# database_manager.py
import sqlite3
import pandas as pd

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    return conn

def save_prediction(conn, week, demand, waste):
    query = "INSERT INTO predictions (week, demand, waste) VALUES (?, ?, ?)"
    conn.execute(query, (week, demand, waste))
    conn.commit()
