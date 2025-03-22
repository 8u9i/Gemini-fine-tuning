"""
README for Gemini Model Tuning with Database Integration

This project demonstrates how to integrate a database with Google's Gemini API for model tuning.
Instead of using hardcoded training examples, this implementation retrieves examples from a
SQLite database, allowing for more scalable and maintainable training data management.

## Files in this project:

1. db_setup.py - Creates and populates a SQLite database with training examples
2. test_db_connection.py - Tests the database connection and displays sample data
3. gemini_tuning_with_db.py - Basic implementation of Gemini model tuning with database integration
4. advanced_gemini_tuning_with_db.py - Comprehensive implementation with additional features

## Setup Instructions:

1. Install the required dependencies:
   ```
   pip install google-generativeai
   ```

2. Set up your Google API key as an environment variable:
   ```
   export GOOGLE_API_KEY="your-api-key"
   ```

3. Create and populate the database:
   ```
   python db_setup.py
   ```

4. Verify the database setup:
   ```
   python test_db_connection.py
   ```

5. Run the model tuning script:
   ```
   python gemini_tuning_with_db.py
   ```
   
   Or for the advanced implementation:
   ```
   python advanced_gemini_tuning_with_db.py
   ```

## Implementation Details:

The implementation follows these key steps:

1. Database Setup:
   - Creates a SQLite database with a table for training examples
   - Populates the table with sample input-output pairs

2. Database Integration:
   - Connects to the database and retrieves training examples
   - Converts the examples to the format required by the Gemini API

3. Model Tuning:
   - Initializes the Gemini client with your API key
   - Creates a TuningDataset from the database examples
   - Starts a tuning job with the specified configuration
   - Tests the tuned model with sample inputs

The advanced implementation includes additional features:
   - Database connection pooling
   - Batch processing for large datasets
   - Configuration management
   - Logging and monitoring
   - Model evaluation

## Customization:

To use your own training data, you can:
1. Modify the sample_data in db_setup.py
2. Create a script to import data from CSV/JSON files into the database
3. Implement a data collection pipeline that feeds into the database

To customize the tuning process, adjust the configuration parameters in the scripts:
- base_model: The base model to tune
- epoch_count: Number of training epochs
- batch_size: Batch size for training
- learning_rate: Learning rate for training
"""
