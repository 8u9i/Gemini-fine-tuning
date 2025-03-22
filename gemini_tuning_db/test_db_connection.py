"""
Example script to test the database setup and connection
"""
import os
import sqlite3

def test_database_connection():
    """Test the database connection and print sample data."""
    db_path = "data/training_examples.db"
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}. Please run db_setup.py first.")
        return False
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the training_examples table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_examples'")
        if not cursor.fetchone():
            print("training_examples table not found in database")
            conn.close()
            return False
        
        # Get count of examples
        cursor.execute("SELECT COUNT(*) FROM training_examples")
        count = cursor.fetchone()[0]
        print(f"Database contains {count} training examples")
        
        # Get sample data
        cursor.execute("SELECT input_text, output_text FROM training_examples LIMIT 5")
        samples = cursor.fetchall()
        print("\nSample training examples:")
        for i, (input_text, output_text) in enumerate(samples, 1):
            print(f"{i}. Input: '{input_text}' → Output: '{output_text}'")
        
        # Close the connection
        conn.close()
        print("\nDatabase connection test successful!")
        return True
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_database_connection()
