"""
Example script to import training data from a CSV file into the database
"""
import os
import csv
import sqlite3

def import_csv_to_database(csv_file_path, db_path="data/training_examples.db"):
    """
    Import training examples from a CSV file into the SQLite database.
    
    Args:
        csv_file_path: Path to the CSV file containing training examples
        db_path: Path to the SQLite database file
    
    Returns:
        Number of examples imported
    """
    # Check if CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"CSV file not found at {csv_file_path}")
        return 0
    
    # Create database directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            output_text TEXT NOT NULL
        )
        ''')
        
        # Read CSV file and insert data
        count = 0
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            # Skip header if it exists
            header = next(reader, None)
            
            for row in reader:
                if len(row) >= 2:  # Ensure row has at least input and output columns
                    input_text = row[0].strip()
                    output_text = row[1].strip()
                    
                    if input_text and output_text:  # Skip empty entries
                        cursor.execute(
                            "INSERT INTO training_examples (input_text, output_text) VALUES (?, ?)",
                            (input_text, output_text)
                        )
                        count += 1
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        print(f"Successfully imported {count} training examples from {csv_file_path}")
        return count
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 0

def create_sample_csv(csv_file_path):
    """Create a sample CSV file with training examples."""
    sample_data = [
        ["input", "output"],  # Header
        ["1", "2"],
        ["3", "4"],
        ["-3", "-2"],
        ["twenty two", "twenty three"],
        ["two hundred", "two hundred one"],
        ["ninety nine", "one hundred"],
        ["8", "9"],
        ["-98", "-97"],
        ["1,000", "1,001"],
        ["10,100,000", "10,100,001"],
        ["thirteen", "fourteen"],
        ["eighty", "eighty one"],
        ["one", "two"],
        ["three", "four"],
        ["seven", "eight"],
    ]
    
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sample_data)
    
    print(f"Created sample CSV file at {csv_file_path} with {len(sample_data)-1} examples")

if __name__ == "__main__":
    # Create a sample CSV file
    csv_file_path = "data/sample_training_data.csv"
    create_sample_csv(csv_file_path)
    
    # Import the CSV data into the database
    import_csv_to_database(csv_file_path)
