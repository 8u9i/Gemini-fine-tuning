"""
Gemini Model Tuning with Database Integration

This script demonstrates how to use a database to store and retrieve training examples
for fine-tuning a Gemini model, rather than using hardcoded examples.
"""
import os
import sqlite3
from google import genai
from google.genai import types

# Function to connect to the database and retrieve training examples
def get_training_examples_from_db(db_path):
    """
    Retrieve training examples from SQLite database.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        List of tuples containing (input_text, output_text) pairs
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query all training examples
        cursor.execute("SELECT input_text, output_text FROM training_examples")
        examples = cursor.fetchall()
        
        # Close the connection
        conn.close()
        
        print(f"Successfully retrieved {len(examples)} training examples from database.")
        return examples
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

def main():
    # Set up the API key
    # You should set this as an environment variable in production
    # os.environ["GOOGLE_API_KEY"] = "your-api-key"
    
    # Initialize the Gemini client
    try:
        client = genai.Client()  # Gets the key from the GOOGLE_API_KEY env variable
        print("Successfully initialized Gemini client.")
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return
    
    # Database path
    db_path = "data/training_examples.db"
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}. Please run db_setup.py first.")
        return
    
    # Get training examples from database
    training_data = get_training_examples_from_db(db_path)
    
    if not training_data:
        print("No training examples found in database. Exiting.")
        return
    
    # Convert database examples to Gemini TuningDataset format
    try:
        training_dataset = types.TuningDataset(
            examples=[
                types.TuningExample(
                    text_input=input_text,
                    output=output_text,
                )
                for input_text, output_text in training_data
            ],
        )
        print("Successfully created TuningDataset from database examples.")
    except Exception as e:
        print(f"Error creating TuningDataset: {e}")
        return
    
    # Start the tuning job
    try:
        print("Starting model tuning job...")
        tuning_job = client.tunings.tune(
            base_model='models/gemini-1.5-flash-001-tuning',
            training_dataset=training_dataset,
            config=types.CreateTuningJobConfig(
                epoch_count=5,
                batch_size=4,
                learning_rate=0.001,
                tuned_model_display_name="database_tuned_model"
            )
        )
        print(f"Tuning job started successfully. Tuned model: {tuning_job.tuned_model.model}")
    except Exception as e:
        print(f"Error starting tuning job: {e}")
        return
    
    # Test the tuned model
    try:
        print("Testing the tuned model...")
        response = client.models.generate_content(
            model=tuning_job.tuned_model.model,
            contents='five',
        )
        print(f"Model response: {response.text}")
    except Exception as e:
        print(f"Error testing tuned model: {e}")
        return

if __name__ == "__main__":
    main()
