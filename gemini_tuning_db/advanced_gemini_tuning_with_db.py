"""
Advanced Gemini Model Tuning with Database Integration

This script demonstrates a more comprehensive implementation of Gemini model tuning
with database integration, including:
- Database connection pooling
- Batch processing for large datasets
- Configuration management
- Logging and monitoring
- Model evaluation
"""
import os
import sqlite3
import logging
import json
import time
from datetime import datetime
from google import genai
from google.genai import types

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gemini_tuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gemini_tuning")

class DatabaseManager:
    """Manages database connections and operations for training data."""
    
    def __init__(self, db_path):
        """Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._check_database()
    
    def _check_database(self):
        """Check if the database exists and has the expected schema."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at {self.db_path}. Please run db_setup.py first.")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if the training_examples table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_examples'")
            if not cursor.fetchone():
                raise Exception("training_examples table not found in database")
            
            conn.close()
        except sqlite3.Error as e:
            raise Exception(f"Database validation error: {e}")
    
    def get_training_examples(self, limit=None, offset=0):
        """Retrieve training examples from the database.
        
        Args:
            limit: Maximum number of examples to retrieve (None for all)
            offset: Number of examples to skip
            
        Returns:
            List of tuples containing (input_text, output_text) pairs
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if limit is not None:
                cursor.execute(
                    "SELECT input_text, output_text FROM training_examples LIMIT ? OFFSET ?", 
                    (limit, offset)
                )
            else:
                cursor.execute("SELECT input_text, output_text FROM training_examples")
                
            examples = cursor.fetchall()
            conn.close()
            
            logger.info(f"Retrieved {len(examples)} training examples from database")
            return examples
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise
    
    def get_example_count(self):
        """Get the total number of training examples in the database.
        
        Returns:
            Integer count of examples
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM training_examples")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except sqlite3.Error as e:
            logger.error(f"Database error when counting examples: {e}")
            raise

    def add_training_example(self, input_text, output_text):
        """Add a new training example to the database.
        
        Args:
            input_text: The input text for the example
            output_text: The expected output text
            
        Returns:
            ID of the inserted example
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO training_examples (input_text, output_text) VALUES (?, ?)",
                (input_text, output_text)
            )
            example_id = cursor.lastrowid
            conn.commit()
            conn.close()
            logger.info(f"Added new training example with ID {example_id}")
            return example_id
        except sqlite3.Error as e:
            logger.error(f"Database error when adding example: {e}")
            raise

class GeminiTuningManager:
    """Manages the Gemini model tuning process."""
    
    def __init__(self, db_manager, config=None):
        """Initialize the tuning manager.
        
        Args:
            db_manager: DatabaseManager instance
            config: Configuration dictionary (or None for defaults)
        """
        self.db_manager = db_manager
        self.config = config or self._get_default_config()
        self.client = None
        self._initialize_client()
    
    def _get_default_config(self):
        """Get default configuration values."""
        return {
            "base_model": "models/gemini-1.5-flash-001-tuning",
            "batch_size": 4,
            "epoch_count": 5,
            "learning_rate": 0.001,
            "tuned_model_display_name": f"db_tuned_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "processing_batch_size": 100  # Number of examples to process at once from DB
        }
    
    def _initialize_client(self):
        """Initialize the Gemini API client."""
        try:
            self.client = genai.Client()  # Gets the key from the GOOGLE_API_KEY env variable
            logger.info("Successfully initialized Gemini client")
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")
            raise
    
    def create_tuning_dataset(self, examples):
        """Create a TuningDataset from examples.
        
        Args:
            examples: List of (input_text, output_text) tuples
            
        Returns:
            TuningDataset object
        """
        try:
            dataset = types.TuningDataset(
                examples=[
                    types.TuningExample(
                        text_input=input_text,
                        output=output_text,
                    )
                    for input_text, output_text in examples
                ],
            )
            logger.info(f"Created TuningDataset with {len(examples)} examples")
            return dataset
        except Exception as e:
            logger.error(f"Error creating TuningDataset: {e}")
            raise
    
    def tune_model(self):
        """Tune the model using examples from the database.
        
        Returns:
            Tuning job object
        """
        # Get total count of examples
        total_examples = self.db_manager.get_example_count()
        logger.info(f"Starting model tuning with {total_examples} total examples")
        
        if total_examples == 0:
            raise ValueError("No training examples found in database")
        
        # For small datasets, process all at once
        if total_examples <= self.config["processing_batch_size"]:
            examples = self.db_manager.get_training_examples()
            training_dataset = self.create_tuning_dataset(examples)
        else:
            # For larger datasets, process in batches
            logger.info(f"Processing large dataset in batches of {self.config['processing_batch_size']}")
            all_examples = []
            offset = 0
            
            while offset < total_examples:
                batch = self.db_manager.get_training_examples(
                    limit=self.config["processing_batch_size"], 
                    offset=offset
                )
                all_examples.extend(batch)
                offset += len(batch)
                logger.info(f"Processed {offset}/{total_examples} examples")
            
            training_dataset = self.create_tuning_dataset(all_examples)
        
        # Start the tuning job
        try:
            logger.info("Starting model tuning job...")
            start_time = time.time()
            
            tuning_job = self.client.tunings.tune(
                base_model=self.config["base_model"],
                training_dataset=training_dataset,
                config=types.CreateTuningJobConfig(
                    epoch_count=self.config["epoch_count"],
                    batch_size=self.config["batch_size"],
                    learning_rate=self.config["learning_rate"],
                    tuned_model_display_name=self.config["tuned_model_display_name"]
                )
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Tuning job started successfully in {elapsed_time:.2f} seconds")
            logger.info(f"Tuned model: {tuning_job.tuned_model.model}")
            
            return tuning_job
        except Exception as e:
            logger.error(f"Error starting tuning job: {e}")
            raise
    
    def test_model(self, model_name, test_inputs):
        """Test the tuned model with sample inputs.
        
        Args:
            model_name: Name of the tuned model
            test_inputs: List of input strings to test
            
        Returns:
            Dictionary mapping inputs to model responses
        """
        results = {}
        
        for test_input in test_inputs:
            try:
                logger.info(f"Testing model with input: '{test_input}'")
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=test_input,
                )
                results[test_input] = response.text
                logger.info(f"Model response for '{test_input}': '{response.text}'")
            except Exception as e:
                logger.error(f"Error testing model with input '{test_input}': {e}")
                results[test_input] = f"ERROR: {str(e)}"
        
        return results
    
    def list_tuned_models(self):
        """List all available tuned models.
        
        Returns:
            List of model information objects
        """
        try:
            models = list(self.client.models.list())
            tuned_models = [model for model in models if "tuned" in model.name.lower()]
            logger.info(f"Found {len(tuned_models)} tuned models")
            return tuned_models
        except Exception as e:
            logger.error(f"Error listing tuned models: {e}")
            raise

def main():
    """Main function to run the Gemini tuning process."""
    # Configuration
    config = {
        "base_model": "models/gemini-1.5-flash-001-tuning",
        "batch_size": 4,
        "epoch_count": 5,
        "learning_rate": 0.001,
        "tuned_model_display_name": f"db_tuned_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "processing_batch_size": 100
    }
    
    # Database path
    db_path = "data/training_examples.db"
    
    try:
        # Initialize database manager
        logger.info(f"Initializing database manager with database at {db_path}")
        db_manager = DatabaseManager(db_path)
        
        # Initialize tuning manager
        logger.info("Initializing Gemini tuning manager")
        tuning_manager = GeminiTuningManager(db_manager, config)
        
        # List existing tuned models
        logger.info("Listing existing tuned models")
        existing_models = tuning_manager.list_tuned_models()
        for model in existing_models:
            logger.info(f"Found tuned model: {model.name}")
        
        # Tune the model
        logger.info("Starting model tuning process")
        tuning_job = tuning_manager.tune_model()
        
        # Test the tuned model
        test_inputs = ["five", "ninety", "two hundred fifty", "-10"]
        logger.info(f"Testing tuned model with {len(test_inputs)} inputs")
        test_results = tuning_manager.test_model(tuning_job.tuned_model.model, test_inputs)
        
        # Save test results
        results_file = "tuning_test_results.json"
        with open(results_file, "w") as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"Test results saved to {results_file}")
        
        logger.info("Gemini model tuning with database integration completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
