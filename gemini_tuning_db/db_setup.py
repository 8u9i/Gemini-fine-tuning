"""
Database setup script for Gemini model tuning.
This script creates a SQLite database with a table for training examples.
"""
import sqlite3
import os

# Create database directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Connect to SQLite database (will be created if it doesn't exist)
conn = sqlite3.connect('data/training_examples.db')
cursor = conn.cursor()

# Create table for training examples
cursor.execute('''
CREATE TABLE IF NOT EXISTS training_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_text TEXT NOT NULL,
    output_text TEXT NOT NULL
)
''')

# Sample data - number sequence examples
sample_data = [
    ("1", "2"),
    ("3", "4"),
    ("-3", "-2"),
    ("twenty two", "twenty three"),
    ("two hundred", "two hundred one"),
    ("ninety nine", "one hundred"),
    ("8", "9"),
    ("-98", "-97"),
    ("1,000", "1,001"),
    ("10,100,000", "10,100,001"),
    ("thirteen", "fourteen"),
    ("eighty", "eighty one"),
    ("one", "two"),
    ("three", "four"),
    ("seven", "eight"),
]

# Insert sample data
cursor.executemany(
    "INSERT INTO training_examples (input_text, output_text) VALUES (?, ?)",
    sample_data
)

# Commit changes and close connection
conn.commit()
print(f"Database created with {len(sample_data)} training examples.")
conn.close()
