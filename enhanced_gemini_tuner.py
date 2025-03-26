#!/usr/bin/env python3
"""
Enhanced Gemini Tuner

A user-friendly tool for processing text files and tuning Google's Gemini models.
Features both command-line and graphical interfaces with interactive workflows.

Usage:
    python gemini_tuner.py [OPTIONS] COMMAND [ARGS]...
    python gemini_tuner.py --gui (launches graphical interface)
    python gemini_tuner.py --interactive (launches interactive CLI mode)

Requirements:
    - Google AI Python SDK
    - python-dotenv (for .env file support)
    - tqdm (for progress bars)
    - PySimpleGUI (for graphical interface)
    - Valid Google AI API key
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Generator
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('GeminiTuner')

# Will be imported if available, otherwise handled in requirements check
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    genai = None
    types = None
    HAS_GENAI = False

# Import dotenv for environment variable support
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    load_dotenv = None
    HAS_DOTENV = False

# Import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    tqdm = None
    HAS_TQDM = False

# Import PySimpleGUI for GUI
try:
    import PySimpleGUI as sg
    HAS_GUI = True
except ImportError:
    sg = None
    HAS_GUI = False


class Config:
    """Configuration management for Gemini Tuner."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (.env)
        """
        self.config_file = config_file
        self.config = {}
        self.load_config()
        
    def load_config(self) -> None:
        """Load configuration from environment variables and .env file."""
        # First try to load from .env file if dotenv is available
        if HAS_DOTENV:
            if self.config_file and os.path.isfile(self.config_file):
                load_dotenv(self.config_file)
                logger.info(f"Loaded configuration from {self.config_file}")
            else:
                # Look for .env file in current directory and parent directories
                env_file = self._find_env_file()
                if env_file:
                    load_dotenv(env_file)
                    self.config_file = env_file
                    logger.info(f"Loaded configuration from {env_file}")
        
        # Load configuration from environment variables
        self.config = {
            "api_key": os.environ.get("GOOGLE_API_KEY", ""),
            "model_name": os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
            "epochs": int(os.environ.get("GEMINI_EPOCHS", "5")),
            "batch_size": int(os.environ.get("GEMINI_BATCH_SIZE", "4")),
            "learning_rate": float(os.environ.get("GEMINI_LEARNING_RATE", "1.0")),
            "delimiter": os.environ.get("EXAMPLE_DELIMITER", "---"),
            "input_prefix": os.environ.get("INPUT_PREFIX", "INPUT:"),
            "output_prefix": os.environ.get("OUTPUT_PREFIX", "OUTPUT:"),
        }
    
    def save_config(self, file_path: Optional[str] = None) -> None:
        """
        Save configuration to a file.
        
        Args:
            file_path: Path to save the configuration file
        """
        if not file_path:
            file_path = self.config_file or ".env"
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Gemini Tuner Configuration - Created {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"# Google AI API key\n")
            f.write(f"GOOGLE_API_KEY={self.config['api_key']}\n\n")
            f.write(f"# Model settings\n")
            f.write(f"GEMINI_MODEL={self.config['model_name']}\n")
            f.write(f"GEMINI_EPOCHS={self.config['epochs']}\n")
            f.write(f"GEMINI_BATCH_SIZE={self.config['batch_size']}\n")
            f.write(f"GEMINI_LEARNING_RATE={self.config['learning_rate']}\n\n")
            f.write(f"# File format settings\n")
            f.write(f"EXAMPLE_DELIMITER={self.config['delimiter']}\n")
            f.write(f"INPUT_PREFIX={self.config['input_prefix']}\n")
            f.write(f"OUTPUT_PREFIX={self.config['output_prefix']}\n")
            
        logger.info(f"Configuration saved to {file_path}")
        self.config_file = file_path
    
    def update(self, **kwargs) -> None:
        """
        Update configuration with new values.
        
        Args:
            **kwargs: Configuration key-value pairs
        """
        self.config.update(kwargs)
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
        
    def _find_env_file(self, max_depth: int = 3) -> Optional[str]:
        """
        Find .env file in current directory or parent directories.
        
        Args:
            max_depth: Maximum number of parent directories to check
            
        Returns:
            Path to .env file if found, None otherwise
        """
        current_dir = os.getcwd()
        for _ in range(max_depth + 1):
            env_file = os.path.join(current_dir, '.env')
            if os.path.isfile(env_file):
                return env_file
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Reached root directory
                break
            current_dir = parent_dir
        return None
    
    @staticmethod
    def create_sample_config(file_path: str = '.env.example') -> None:
        """
        Create a sample configuration file.
        
        Args:
            file_path: Path to save the sample configuration file
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("""# Gemini Tuner Configuration Example

# Google AI API key (required for model tuning)
GOOGLE_API_KEY=your_api_key_here

# Default model to use for tuning (optional)
GEMINI_MODEL=gemini-2.0-flash

# Default settings for tuning (optional)
GEMINI_EPOCHS=5
GEMINI_BATCH_SIZE=4
GEMINI_LEARNING_RATE=1.0

# Default file format settings (optional)
EXAMPLE_DELIMITER=---
INPUT_PREFIX=INPUT:
OUTPUT_PREFIX=OUTPUT:
""")
        logger.info(f"Sample configuration file created at {file_path}")
        logger.info("Copy this file to .env and update with your actual values.")


class DataProcessor:
    """Process text files into training data for Gemini model tuning."""
    
    def __init__(self, config: Config):
        """
        Initialize data processor.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        
    def process_file(self, 
                    file_path: str, 
                    delimiter: Optional[str] = None, 
                    input_prefix: Optional[str] = None, 
                    output_prefix: Optional[str] = None,
                    max_input_chars: int = 40000,
                    max_output_chars: int = 5000) -> List[Dict[str, str]]:
        """
        Process a text file into a format suitable for Gemini model tuning.
        
        Args:
            file_path: Path to the text file
            delimiter: String that separates examples in the file
            input_prefix: Prefix that marks the beginning of input text
            output_prefix: Prefix that marks the beginning of output text
            max_input_chars: Maximum allowed characters for input (Gemini limit: 40,000)
            max_output_chars: Maximum allowed characters for output (Gemini limit: 5,000)
            
        Returns:
            List of dictionaries with 'text_input' and 'output' keys
        """
        # Use configuration values if not provided
        delimiter = delimiter or self.config.get('delimiter')
        input_prefix = input_prefix or self.config.get('input_prefix')
        output_prefix = output_prefix or self.config.get('output_prefix')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split the content by delimiter
        examples = content.split(delimiter)
        
        # Process each example
        training_data = []
        valid_count = 0
        invalid_count = 0
        
        # Use tqdm for progress bar if available
        example_iterator = tqdm(enumerate(examples), total=len(examples), desc="Processing examples") if HAS_TQDM else enumerate(examples)
        
        for i, example in example_iterator:
            example = example.strip()
            if not example:
                continue
                
            # Find input and output sections
            input_start = example.find(input_prefix)
            output_start = example.find(output_prefix)
            
            if input_start == -1 or output_start == -1:
                logger.warning(f"Example {i+1} does not contain both input and output prefixes. Skipping.")
                invalid_count += 1
                continue
                
            # Extract input and output text
            input_text = example[input_start + len(input_prefix):output_start].strip()
            output_text = example[output_start + len(output_prefix):].strip()
            
            # Validate against character limits
            if len(input_text) > max_input_chars:
                logger.warning(f"Input text in example {i+1} exceeds {max_input_chars} characters. Truncating.")
                input_text = input_text[:max_input_chars]
                
            if len(output_text) > max_output_chars:
                logger.warning(f"Output text in example {i+1} exceeds {max_output_chars} characters. Truncating.")
                output_text = output_text[:max_output_chars]
                
            # Add to training data
            training_data.append({
                "text_input": input_text,
                "output": output_text
            })
            valid_count += 1
            
        logger.info(f"Processed {len(training_data)} examples from {file_path}")
        logger.info(f"Valid examples: {valid_count}, Invalid examples: {invalid_count}")
        return training_data
    
    def process_multiple_files(self, file_paths: List[str], **kwargs) -> List[Dict[str, str]]:
        """
        Process multiple text files into a single training dataset.
        
        Args:
            file_paths: List of paths to text files
            **kwargs: Additional arguments for process_file
            
        Returns:
            Combined list of training examples
        """
        all_training_data = []
        
        for file_path in file_paths:
            try:
                training_data = self.process_file(file_path, **kwargs)
                all_training_data.extend(training_data)
                logger.info(f"Added {len(training_data)} examples from {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                
        logger.info(f"Total examples in combined dataset: {len(all_training_data)}")
        return all_training_data
    
    def save_training_data(self, training_data: List[Dict[str, str]], output_file: str) -> None:
        """
        Save the processed training data to a JSON file.
        
        Args:
            training_data: List of dictionaries with 'text_input' and 'output' keys
            output_file: Path to save the JSON file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2)
            
        logger.info(f"Training data saved to {output_file}")
    
    def load_training_data(self, input_file: str) -> List[Dict[str, str]]:
        """
        Load training data from a JSON file.
        
        Args:
            input_file: Path to the JSON file
            
        Returns:
            List of dictionaries with 'text_input' and 'output' keys
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
            
        logger.info(f"Loaded {len(training_data)} examples from {input_file}")
        return training_data
    
    def preview_training_data(self, training_data: List[Dict[str, str]], num_examples: int = 2) -> None:
        """
        Preview training data examples.
        
        Args:
            training_data: List of dictionaries with 'text_input' and 'output' keys
            num_examples: Number of examples to preview
        """
        print("\nTraining Data Preview:")
        print("======================")
        
        for i, example in enumerate(training_data[:num_examples]):
            print(f"\nExample {i+1}:")
            print(f"Input: \"{example['text_input'][:100]}{'...' if len(example['text_input']) > 100 else ''}\"")
            print(f"Output: \"{example['output'][:100]}{'...' if len(example['output']) > 100 else ''}\"")
            
        if len(training_data) > num_examples:
            print(f"\n... and {len(training_data) - num_examples} more examples")


# Import the ChatManager class
from chat_manager import ChatManager

class ModelManager:
    """Manage Gemini model tuning and inference."""
    
    def __init__(self, config: Config):
        """
        Initialize model manager.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.client = None
        self.chat_manager = None
        self._initialize_genai()
        
    def _initialize_genai(self) -> bool:
        """
        Initialize the Google Generative AI SDK.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        # Set default values for client and chat_manager
        self.client = None
        self.chat_manager = None
        
        if not HAS_GENAI:
            logger.error("Google Generative AI SDK not found. Model tuning functionality will be limited.")
            return False
            
        api_key = self.config.get('api_key')
        if api_key:
            try:
                # Initialize client with API key
                self.client = genai.Client(api_key=api_key)
                
                # Initialize chat manager
                self.chat_manager = ChatManager(self)
                
                # Test API key validity
                models = self.client.list_models()
                tunable_models = [m.name for m in models if hasattr(m, 'supported_generation_methods') and 'tuning' in m.supported_generation_methods]
                logger.info(f"Successfully authenticated with Google AI API.")
                logger.info(f"Available models for tuning: {tunable_models}")
                return True
            except Exception as e:
                logger.error(f"Error authenticating with Google AI API: {e}")
                logger.warning("Continuing in limited mode. You can still process files but not train models.")
                self.client = None
                self.chat_manager = None
                return False
        else:
            logger.warning("No API key provided. You can still process files but not train models.")
            return False
    
    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate Google AI API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if API key is valid, False otherwise
        """
        if not HAS_GENAI:
            logger.error("Google Generative AI SDK not found. Cannot validate API key.")
            return False
            
        try:
            # Create a temporary client to validate the API key
            temp_client = genai.Client(api_key=api_key)
            models = temp_client.list_models()
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
    
    def train_model(self, 
                   training_data: List[Dict[str, str]], 
                   model_name: Optional[str] = None,
                   display_name: Optional[str] = None,
                   epochs: Optional[int] = None,
                   batch_size: Optional[int] = None,
                   learning_rate_multiplier: Optional[float] = None) -> Optional[str]:
        """
        Train a Gemini model using the provided training data.
        
        Args:
            training_data: List of dictionaries with 'text_input' and 'output' keys
            model_name: Base model to tune
            display_name: Display name for the tuned model
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate_multiplier: Learning rate multiplier
            
        Returns:
            Tuned model ID if successful, None otherwise
        """
        if not HAS_GENAI:
            logger.error("Google Generative AI SDK not available. Cannot train model.")
            return None
            
        if not self.config.get('api_key'):
            logger.error("API key not provided. Cannot train model.")
            return None
            
        if not training_data:
            logger.error("No training data provided. Cannot train model.")
            return None
            
        # Use configuration values if not provided
        model_name = model_name or self.config.get('model_name')
        display_name = display_name or f"Tuned Model - {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        epochs = epochs or self.config.get('epochs')
        batch_size = batch_size or self.config.get('batch_size')
        learning_rate_multiplier = learning_rate_multiplier or self.config.get('learning_rate')
            
        try:
            # Create a tuning job
            logger.info(f"Starting model tuning with {len(training_data)} examples...")
            logger.info(f"Base model: {model_name}")
            logger.info(f"Training configuration: epochs={epochs}, batch_size={batch_size}, learning_rate_multiplier={learning_rate_multiplier}")
            
            # Use the client instance to create tuned model
            tuning_job = self.client.create_tuned_model(
                source_model=model_name,
                training_data=training_data,
                display_name=display_name,
                tuning_task=types.TuningTask(
                    hyperparameters=types.TuningHyperparameters(
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate_multiplier=learning_rate_multiplier
                    )
                )
            )
            
            # Wait for the tuning job to complete
            logger.info("Tuning job started. This may take a while...")
            
            # If tqdm is available, show a progress bar
            if HAS_TQDM:
                with tqdm(total=100, desc="Training model") as pbar:
                    last_progress = 0
                    while True:
                        # Check job status
                        job_status = tuning_job.get_status()
                        if job_status.state == "SUCCEEDED":
                            pbar.update(100 - last_progress)
                            break
                        elif job_status.state == "FAILED":
                            logger.error(f"Tuning job failed: {job_status.error}")
                            return None
                        
                        # Update progress bar if progress information is available
                        if hasattr(job_status, 'progress'):
                            progress = int(job_status.progress * 100)
                            pbar.update(progress - last_progress)
                            last_progress = progress
                        
                        time.sleep(10)  # Check status every 10 seconds
            else:
                # Simple waiting without progress bar
                while True:
                    job_status = tuning_job.get_status()
                    if job_status.state == "SUCCEEDED":
                        break
                    elif job_status.state == "FAILED":
                        logger.error(f"Tuning job failed: {job_status.error}")
                        return None
                    
                    logger.info("Training in progress...")
                    time.sleep(30)  # Check status every 30 seconds
            
            tuned_model = tuning_job.result()
            
            logger.info(f"Model tuning completed successfully!")
            logger.info(f"Tuned model ID: {tuned_model.name}")
            return tuned_model.name
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all tuned models.
        
        Returns:
            List of dictionaries with model information
        """
        if not HAS_GENAI or not self.client:
            logger.error("Cannot list models without Google AI SDK and valid API key.")
            return []
            
        try:
            models = self.client.list_tuned_models()
            model_info = []
            
            for model in models:
                info = {
                    "name": model.display_name,
                    "id": model.name,
                    "base_model": model.source_model,
                    "created": model.create_time,
                    "state": model.state
                }
                model_info.append(info)
                
            return model_info
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a tuned model.
        
        Args:
            model_id: ID of the tuned model
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not HAS_GENAI or not self.client:
            logger.error("Cannot delete model without Google AI SDK and valid API key.")
            return False
            
        try:
            self.client.delete_tuned_model(model_id)
            logger.info(f"Model {model_id} deleted successfully.")
            return True
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False
    
    def test_model(self, model_id: str, test_input: str) -> Optional[str]:
        """
        Test a tuned model with a sample input.
        
        Args:
            model_id: ID of the tuned model
            test_input: Input text to test the model with
            
        Returns:
            Model response if successful, None otherwise
        """
        if not HAS_GENAI or not self.client:
            logger.error("Cannot test model without Google AI SDK and valid API key.")
            return None
            
        try:
            # Get the tuned model
            model = self.client.get_tuned_model(model_id)
            
            # Generate content using the client's models interface with configuration parameters
            generation_config = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1024,
                top_p=0.95,
                top_k=40
            )
            
            response = self.client.models.generate_content(
                model=model.name,
                contents=test_input,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error testing model: {e}")
            return None
            
    def create_chat(self, model_name: str = None) -> Optional[str]:
        """
        Create a new chat session.
        
        Args:
            model_name: Name of the model to use for chat (defaults to configured model)
            
        Returns:
            Chat ID if successful, None otherwise
        """
        if not self.chat_manager:
            logger.error("Chat functionality not available. Please initialize with a valid API key.")
            return None
            
        return self.chat_manager.create_chat(model_name)
        
    def send_message(self, chat_id: str, message: str, stream: bool = False) -> Union[str, Generator]:
        """
        Send a message to a chat session.
        
        Args:
            chat_id: ID of the chat session
            message: Message to send
            stream: Whether to stream the response
            
        Returns:
            Response text or generator for streaming responses
        """
        if not self.chat_manager:
            logger.error("Chat functionality not available. Please initialize with a valid API key.")
            return None
            
        return self.chat_manager.send_message(chat_id, message, stream)
        
    def get_chat_history(self, chat_id: str) -> List[Dict[str, str]]:
        """
        Get the history of a chat session.
        
        Args:
            chat_id: ID of the chat session
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        if not self.chat_manager:
            logger.error("Chat functionality not available. Please initialize with a valid API key.")
            return []
            
        return self.chat_manager.get_history(chat_id)
        
    def list_chats(self) -> List[Dict[str, Any]]:
        """
        List all active chat sessions.
        
        Returns:
            List of dictionaries with chat information
        """
        if not self.chat_manager:
            logger.error("Chat functionality not available. Please initialize with a valid API key.")
            return []
            
        return self.chat_manager.list_chats()
        
    def delete_chat(self, chat_id: str) -> bool:
        """
        Delete a chat session.
        
        Args:
            chat_id: ID of the chat session
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self.chat_manager:
            logger.error("Chat functionality not available. Please initialize with a valid API key.")
            return False
            
        return self.chat_manager.delete_chat(chat_id)


class InteractiveCLI:
    """Interactive command-line interface for Gemini Tuner."""
    
    def __init__(self, config: Config, processor: DataProcessor, model_manager: ModelManager):
        """
        Initialize interactive CLI.
        
        Args:
            config: Configuration manager
            processor: Data processor
            model_manager: Model manager
        """
        self.config = config
        self.processor = processor
        self.model_manager = model_manager
        
    def run(self) -> None:
        """Run the interactive CLI."""
        self._print_header()
        
        # Step 1: Configuration
        self._setup_configuration()
        
        # Step 2: Input file selection
        file_paths = self._select_input_files()
        if not file_paths:
            logger.error("No input files selected. Exiting.")
            return
            
        # Step 3: Processing options
        delimiter = self._prompt("Example delimiter", self.config.get('delimiter'))
        input_prefix = self._prompt("Input prefix", self.config.get('input_prefix'))
        output_prefix = self._prompt("Output prefix", self.config.get('output_prefix'))
        
        # Step 4: Process files
        training_data = []
        try:
            if len(file_paths) == 1:
                training_data = self.processor.process_file(
                    file_paths[0],
                    delimiter=delimiter,
                    input_prefix=input_prefix,
                    output_prefix=output_prefix
                )
            else:
                training_data = self.processor.process_multiple_files(
                    file_paths,
                    delimiter=delimiter,
                    input_prefix=input_prefix,
                    output_prefix=output_prefix
                )
                
            # Preview data
            if self._confirm("Preview processed data?", True):
                self.processor.preview_training_data(training_data)
                
            # Save data
            if self._confirm("Save processed data?", True):
                output_file = self._prompt("Output file path", "training_data.json")
                self.processor.save_training_data(training_data, output_file)
        except Exception as e:
            logger.error(f"Error processing files: {e}")
            return
            
        # Step 5: Model training
        if self._confirm("Train a model with this data?", True):
            model_name = self._prompt("Base model", self.config.get('model_name'))
            display_name = self._prompt("Display name", f"Tuned Model - {datetime.now().strftime('%Y%m%d_%H%M%S')}")
            epochs = int(self._prompt("Number of epochs", str(self.config.get('epochs'))))
            batch_size = int(self._prompt("Batch size", str(self.config.get('batch_size'))))
            learning_rate = float(self._prompt("Learning rate multiplier", str(self.config.get('learning_rate'))))
            
            model_id = self.model_manager.train_model(
                training_data=training_data,
                model_name=model_name,
                display_name=display_name,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate_multiplier=learning_rate
            )
            
            if model_id and self._confirm("Test the model?", True):
                test_input = input("Enter test input: ")
                response = self.model_manager.test_model(model_id, test_input)
                if response:
                    print("\nModel response:")
                    print("==============")
                    print(response)
        
        print("\nThank you for using Gemini Tuner!")
    
    def _print_header(self) -> None:
        """Print the CLI header."""
        print("\n🔮 Gemini Tuner - Interactive Mode 🔮")
        print("-------------------------------------\n")
    
    def _setup_configuration(self) -> None:
        """Set up configuration interactively."""
        print("Step 1/5: Configuration")
        
        # Check if configuration file exists
        if self.config.config_file:
            if not self._confirm(f"Use existing configuration from {self.config.config_file}?", True):
                self._create_new_config()
        else:
            self._create_new_config()
            
        # Validate API key
        api_key = self.config.get('api_key')
        if not api_key:
            api_key = self._prompt_password("Enter your Google AI API key")
            if api_key:
                self.config.update(api_key=api_key)
                if self.model_manager.validate_api_key(api_key):
                    print("API key validated successfully!")
                    if self._confirm("Save this API key to configuration?", True):
                        self.config.save_config()
                else:
                    print("API key validation failed. Continuing in limited mode.")
    
    def _create_new_config(self) -> None:
        """Create a new configuration file interactively."""
        if self._confirm("Create new configuration file?", True):
            config_path = self._prompt("Configuration file path", ".env")
            
            # Get configuration values
            api_key = self._prompt_password("Enter your Google AI API key (leave empty to skip)")
            model_name = self._prompt("Default model name", "gemini-2.0-flash")
            epochs = self._prompt("Default epochs", "5")
            batch_size = self._prompt("Default batch size", "4")
            learning_rate = self._prompt("Default learning rate multiplier", "1.0")
            delimiter = self._prompt("Default example delimiter", "---")
            input_prefix = self._prompt("Default input prefix", "INPUT:")
            output_prefix = self._prompt("Default output prefix", "OUTPUT:")
            
            # Update configuration
            self.config.update(
                api_key=api_key,
                model_name=model_name,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                delimiter=delimiter,
                input_prefix=input_prefix,
                output_prefix=output_prefix
            )
            
            # Save configuration
            self.config.save_config(config_path)
    
    def _select_input_files(self) -> List[str]:
        """
        Select input files interactively.
        
        Returns:
            List of selected file paths
        """
        print("\nStep 2/5: Input File Selection")
        
        file_paths = []
        while True:
            file_path = self._prompt("Enter path to input file (or leave empty to finish)")
            if not file_path:
                break
                
            if os.path.isfile(file_path):
                file_paths.append(file_path)
                print(f"Added file: {file_path}")
            else:
                print(f"File not found: {file_path}")
                
        return file_paths
    
    def _prompt(self, message: str, default: Optional[str] = None) -> str:
        """
        Prompt for user input with optional default value.
        
        Args:
            message: Prompt message
            default: Default value
            
        Returns:
            User input or default value
        """
        if default:
            user_input = input(f"> {message} [{default}]: ")
            return user_input if user_input else default
        else:
            return input(f"> {message}: ")
    
    def _prompt_password(self, message: str) -> str:
        """
        Prompt for password input (without echo).
        
        Args:
            message: Prompt message
            
        Returns:
            Password input
        """
        try:
            import getpass
            return getpass.getpass(f"> {message}: ")
        except ImportError:
            # Fallback if getpass is not available
            return input(f"> {message} (warning: input will be visible): ")
    
    def _confirm(self, message: str, default: bool = False) -> bool:
        """
        Prompt for confirmation.
        
        Args:
            message: Confirmation message
            default: Default value
            
        Returns:
            True if confirmed, False otherwise
        """
        default_str = "Y/n" if default else "y/N"
        response = input(f"> {message} [{default_str}]: ").lower()
        
        if not response:
            return default
        return response.startswith('y')


class GUI:
    """Graphical user interface for Gemini Tuner."""
    
    def __init__(self, config: Config, processor: DataProcessor, model_manager: ModelManager):
        """
        Initialize GUI.
        
        Args:
            config: Configuration manager
            processor: Data processor
            model_manager: Model manager
        """
        self.config = config
        self.processor = processor
        self.model_manager = model_manager
        self.window = None
        self.training_data = []
        
        # Set theme
        if HAS_GUI:
            sg.theme('LightBlue2')
    
    def run(self) -> None:
        """Run the GUI."""
        if not HAS_GUI:
            logger.error("PySimpleGUI not installed. Cannot run GUI.")
            print("Please install PySimpleGUI: pip install PySimpleGUI")
            return
            
        # Create the main window
        self.create_main_window()
        
        # Event loop
        while True:
            event, values = self.window.read()
            
            if event == sg.WINDOW_CLOSED:
                break
                
            # Handle navigation
            if event.startswith('NAV_'):
                self.handle_navigation(event)
                
            # Handle setup events
            elif event.startswith('SETUP_'):
                self.handle_setup_events(event, values)
                
            # Handle process events
            elif event.startswith('PROCESS_'):
                self.handle_process_events(event, values)
                
            # Handle train events
            elif event.startswith('TRAIN_'):
                self.handle_train_events(event, values)
                
            # Handle test events
            elif event.startswith('TEST_'):
                self.handle_test_events(event, values)
                
            # Handle model manager events
            elif event.startswith('MODEL_'):
                self.handle_model_events(event, values)
                
        self.window.close()
    
    def create_main_window(self) -> None:
        """Create the main application window."""
        # Navigation sidebar
        navigation = [
            [sg.Text("Navigation", font=("Helvetica", 14))],
            [sg.Button("Setup", key="NAV_SETUP", size=(15, 1))],
            [sg.Button("Process Files", key="NAV_PROCESS", size=(15, 1))],
            [sg.Button("Train Models", key="NAV_TRAIN", size=(15, 1))],
            [sg.Button("Test Models", key="NAV_TEST", size=(15, 1))],
            [sg.Button("Model Manager", key="NAV_MODELS", size=(15, 1))],
            [sg.HorizontalSeparator()],
            [sg.Text("Recent Projects", font=("Helvetica", 12))],
            [sg.Listbox(values=[], size=(15, 3), key="RECENT_PROJECTS")]
        ]
        
        # Content area with multiple layouts
        setup_layout = self.create_setup_layout()
        process_layout = self.create_process_layout()
        train_layout = self.create_train_layout()
        test_layout = self.create_test_layout()
        model_layout = self.create_model_layout()
        
        # Combine layouts in a single column with visibility switches
        content = [
            [sg.Column(setup_layout, key="CONTENT_SETUP", visible=True)],
            [sg.Column(process_layout, key="CONTENT_PROCESS", visible=False)],
            [sg.Column(train_layout, key="CONTENT_TRAIN", visible=False)],
            [sg.Column(test_layout, key="CONTENT_TEST", visible=False)],
            [sg.Column(model_layout, key="CONTENT_MODELS", visible=False)]
        ]
        
        # Status bar
        status_bar = [
            [sg.Text("Ready", key="STATUS")]
        ]
        
        # Main layout
        layout = [
            [sg.Text("Gemini Tuner", font=("Helvetica", 20), justification="center", expand_x=True)],
            [sg.HorizontalSeparator()],
            [
                sg.Column(navigation, vertical_alignment="top", pad=((0, 10), (10, 10))),
                sg.VerticalSeparator(),
                sg.Column(content, vertical_alignment="top", expand_x=True, expand_y=True, pad=((10, 0), (10, 10)))
            ],
            [sg.HorizontalSeparator()],
            [sg.Column(status_bar, expand_x=True)]
        ]
        
        # Create window
        self.window = sg.Window("Gemini Tuner", layout, size=(800, 600), resizable=True)
        
        # Initialize with API key validation
        self.validate_api_key_status()
    
    def create_setup_layout(self) -> List[List[Any]]:
        """
        Create the setup screen layout.
        
        Returns:
            Layout for the setup screen
        """
        return [
            [sg.Text("Setup", font=("Helvetica", 16))],
            [sg.HorizontalSeparator()],
            
            [sg.Text("API Configuration", font=("Helvetica", 12))],
            [sg.Text("Google AI API Key:"), 
             sg.Input(self.config.get('api_key', ''), key="SETUP_API_KEY", password_char='*'),
             sg.Button("Test", key="SETUP_TEST_API")],
            [sg.Text("API Status: Unknown", key="SETUP_API_STATUS")],
            
            [sg.Text("Default Settings", font=("Helvetica", 12))],
            [sg.Text("Base Model:"), 
             sg.Combo(["gemini-2.0-flash", "gemini-2.0-pro"], 
                     default_value=self.config.get('model_name'),
                     key="SETUP_MODEL")],
            [sg.Text("Training Epochs:"), 
             sg.Input(self.config.get('epochs'), key="SETUP_EPOCHS", size=(5, 1))],
            [sg.Text("Batch Size:"), 
             sg.Input(self.config.get('batch_size'), key="SETUP_BATCH_SIZE", size=(5, 1))],
            [sg.Text("Learning Rate:"), 
             sg.Input(self.config.get('learning_rate'), key="SETUP_LEARNING_RATE", size=(5, 1))],
            
            [sg.Text("File Format Settings", font=("Helvetica", 12))],
            [sg.Text("Example Delimiter:"), 
             sg.Input(self.config.get('delimiter'), key="SETUP_DELIMITER")],
            [sg.Text("Input Prefix:"), 
             sg.Input(self.config.get('input_prefix'), key="SETUP_INPUT_PREFIX")],
            [sg.Text("Output Prefix:"), 
             sg.Input(self.config.get('output_prefix'), key="SETUP_OUTPUT_PREFIX")],
            
            [sg.Button("Save as Default", key="SETUP_SAVE_DEFAULT"), 
             sg.Button("Save as Profile", key="SETUP_SAVE_PROFILE"),
             sg.Button("Reset", key="SETUP_RESET")]
        ]
    
    def create_process_layout(self) -> List[List[Any]]:
        """
        Create the process files screen layout.
        
        Returns:
            Layout for the process files screen
        """
        return [
            [sg.Text("Process Files", font=("Helvetica", 16))],
            [sg.HorizontalSeparator()],
            
            [sg.Text("Input Files", font=("Helvetica", 12))],
            [sg.Button("Add Files", key="PROCESS_ADD_FILES"), 
             sg.Button("Add Directory", key="PROCESS_ADD_DIR")],
            [sg.Listbox(values=[], size=(60, 5), key="PROCESS_FILES")],
            [sg.Button("Remove Selected", key="PROCESS_REMOVE_FILE")],
            
            [sg.Text("Format Settings", font=("Helvetica", 12))],
            [sg.Text("Format:"), 
             sg.Combo(["Default Format"], default_value="Default Format", key="PROCESS_FORMAT")],
            [sg.Text("Example Delimiter:"), 
             sg.Input(self.config.get('delimiter'), key="PROCESS_DELIMITER")],
            [sg.Text("Input Prefix:"), 
             sg.Input(self.config.get('input_prefix'), key="PROCESS_INPUT_PREFIX")],
            [sg.Text("Output Prefix:"), 
             sg.Input(self.config.get('output_prefix'), key="PROCESS_OUTPUT_PREFIX")],
            
            [sg.Text("Output", font=("Helvetica", 12))],
            [sg.Text("Output File:"), 
             sg.Input("training_data.json", key="PROCESS_OUTPUT_FILE"),
             sg.FileSaveAs("Browse", key="PROCESS_BROWSE_OUTPUT")],
            [sg.Checkbox("Preview data before saving", default=True, key="PROCESS_PREVIEW")],
            [sg.Checkbox("Automatically train model after processing", default=False, key="PROCESS_AUTO_TRAIN")],
            
            [sg.Button("Process Files", key="PROCESS_START"), sg.Button("Cancel", key="PROCESS_CANCEL")]
        ]
    
    def create_train_layout(self) -> List[List[Any]]:
        """
        Create the train model screen layout.
        
        Returns:
            Layout for the train model screen
        """
        return [
            [sg.Text("Train Model", font=("Helvetica", 16))],
            [sg.HorizontalSeparator()],
            
            [sg.Text("Training Data", font=("Helvetica", 12))],
            [sg.Text("Data Source:"), 
             sg.Input("training_data.json", key="TRAIN_DATA_SOURCE"),
             sg.FileBrowse("Browse", key="TRAIN_BROWSE_DATA")],
            [sg.Text("Examples: 0  Valid: 0  Invalid: 0", key="TRAIN_DATA_STATS")],
            [sg.Button("View Examples", key="TRAIN_VIEW_EXAMPLES"), 
             sg.Button("Edit Examples", key="TRAIN_EDIT_EXAMPLES")],
            
            [sg.Text("Model Settings", font=("Helvetica", 12))],
            [sg.Text("Base Model:"), 
             sg.Combo(["gemini-2.0-flash", "gemini-2.0-pro"], 
                     default_value=self.config.get('model_name'),
                     key="TRAIN_MODEL")],
            [sg.Text("Display Name:"), 
             sg.Input(f"Tuned Model - {datetime.now().strftime('%Y%m%d_%H%M%S')}", key="TRAIN_DISPLAY_NAME")],
            
            [sg.Text("Advanced Settings:", font=("Helvetica", 12))],
            [sg.Text("Training Epochs:"), 
             sg.Input(self.config.get('epochs'), key="TRAIN_EPOCHS", size=(5, 1))],
            [sg.Text("Batch Size:"), 
             sg.Input(self.config.get('batch_size'), key="TRAIN_BATCH_SIZE", size=(5, 1))],
            [sg.Text("Learning Rate:"), 
             sg.Input(self.config.get('learning_rate'), key="TRAIN_LEARNING_RATE", size=(5, 1))],
            
            [sg.Button("Start Training", key="TRAIN_START"), 
             sg.Button("Save Settings", key="TRAIN_SAVE_SETTINGS"),
             sg.Button("Cancel", key="TRAIN_CANCEL")]
        ]
    
    def create_test_layout(self) -> List[List[Any]]:
        """
        Create the test model screen layout.
        
        Returns:
            Layout for the test model screen
        """
        return [
            [sg.Text("Test Model", font=("Helvetica", 16))],
            [sg.HorizontalSeparator()],
            
            [sg.Text("Model Selection", font=("Helvetica", 12))],
            [sg.Text("Model:"), sg.Combo([], key="TEST_MODEL", size=(40, 1)), 
             sg.Button("Refresh", key="TEST_REFRESH_MODELS")],
            [sg.Text("Base Model: ", key="TEST_BASE_MODEL")],
            [sg.Text("Created: ", key="TEST_CREATED_DATE")],
            
            [sg.Text("Test Input", font=("Helvetica", 12))],
            [sg.Button("Load from File", key="TEST_LOAD_INPUT")],
            [sg.Multiline(size=(60, 5), key="TEST_INPUT")],
            
            [sg.Text("Model Response", font=("Helvetica", 12))],
            [sg.Multiline(size=(60, 10), key="TEST_RESPONSE", disabled=True)],
            
            [sg.Button("Generate Response", key="TEST_GENERATE"), 
             sg.Button("Save Response", key="TEST_SAVE_RESPONSE"),
             sg.Button("Clear", key="TEST_CLEAR")]
        ]
    
    def create_model_layout(self) -> List[List[Any]]:
        """
        Create the model manager screen layout.
        
        Returns:
            Layout for the model manager screen
        """
        return [
            [sg.Text("Model Manager", font=("Helvetica", 16))],
            [sg.HorizontalSeparator()],
            
            [sg.Text("Your Tuned Models", font=("Helvetica", 12))],
            [sg.Table(values=[], headings=["Name", "Created", "Base Model", "State"],
                     auto_size_columns=False, col_widths=[30, 15, 20, 10],
                     justification="left", key="MODEL_TABLE",
                     enable_events=True, select_mode=sg.TABLE_SELECT_MODE_BROWSE)],
            [sg.Button("Refresh List", key="MODEL_REFRESH"), 
             sg.Button("Delete Selected", key="MODEL_DELETE")],
            
            [sg.Text("Model Details", font=("Helvetica", 12))],
            [sg.Text("Name: ", key="MODEL_NAME")],
            [sg.Text("ID: ", key="MODEL_ID")],
            [sg.Text("Base Model: ", key="MODEL_BASE")],
            [sg.Text("Created: ", key="MODEL_CREATED")],
            [sg.Text("State: ", key="MODEL_STATE")],
            
            [sg.Button("Test Model", key="MODEL_TEST"), 
             sg.Button("Export Details", key="MODEL_EXPORT"),
             sg.Button("Clone Settings", key="MODEL_CLONE")]
        ]
    
    def handle_navigation(self, event: str) -> None:
        """
        Handle navigation events.
        
        Args:
            event: Event name
        """
        # Hide all content panels
        for key in ["CONTENT_SETUP", "CONTENT_PROCESS", "CONTENT_TRAIN", "CONTENT_TEST", "CONTENT_MODELS"]:
            self.window[key].update(visible=False)
            
        # Show selected panel
        content_key = f"CONTENT_{event[4:]}"
        self.window[content_key].update(visible=True)
        
        # Update status
        self.window["STATUS"].update(f"Viewing {event[4:].lower()}")
        
        # Special handling for specific panels
        if event == "NAV_SETUP":
            self.validate_api_key_status()
        elif event == "NAV_MODELS":
            self.refresh_model_list()
    
    def handle_setup_events(self, event: str, values: Dict[str, Any]) -> None:
        """
        Handle setup screen events.
        
        Args:
            event: Event name
            values: Form values
        """
        if event == "SETUP_TEST_API":
            api_key = values["SETUP_API_KEY"]
            if api_key:
                self.window["STATUS"].update("Testing API key...")
                if self.model_manager.validate_api_key(api_key):
                    self.window["SETUP_API_STATUS"].update("API Status: Valid", text_color="green")
                    self.window["STATUS"].update("API key is valid")
                    # Update config
                    self.config.update(api_key=api_key)
                else:
                    self.window["SETUP_API_STATUS"].update("API Status: Invalid", text_color="red")
                    self.window["STATUS"].update("API key is invalid")
            else:
                self.window["SETUP_API_STATUS"].update("API Status: No key provided", text_color="orange")
                self.window["STATUS"].update("No API key provided")
                
        elif event == "SETUP_SAVE_DEFAULT":
            try:
                # Update config with form values
                self.config.update(
                    api_key=values["SETUP_API_KEY"],
                    model_name=values["SETUP_MODEL"],
                    epochs=int(values["SETUP_EPOCHS"]),
                    batch_size=int(values["SETUP_BATCH_SIZE"]),
                    learning_rate=float(values["SETUP_LEARNING_RATE"]),
                    delimiter=values["SETUP_DELIMITER"],
                    input_prefix=values["SETUP_INPUT_PREFIX"],
                    output_prefix=values["SETUP_OUTPUT_PREFIX"]
                )
                
                # Save config
                self.config.save_config()
                self.window["STATUS"].update("Configuration saved as default")
                sg.popup("Configuration saved", "Your settings have been saved as the default configuration.")
            except Exception as e:
                self.window["STATUS"].update(f"Error saving configuration: {e}")
                sg.popup_error(f"Error saving configuration: {e}")
                
        elif event == "SETUP_SAVE_PROFILE":
            try:
                # Get profile name
                profile_name = sg.popup_get_text("Enter profile name", "Save Configuration Profile")
                if not profile_name:
                    return
                    
                # Create profile file name
                profile_file = f"{profile_name.lower().replace(' ', '_')}_config.env"
                
                # Update config with form values
                self.config.update(
                    api_key=values["SETUP_API_KEY"],
                    model_name=values["SETUP_MODEL"],
                    epochs=int(values["SETUP_EPOCHS"]),
                    batch_size=int(values["SETUP_BATCH_SIZE"]),
                    learning_rate=float(values["SETUP_LEARNING_RATE"]),
                    delimiter=values["SETUP_DELIMITER"],
                    input_prefix=values["SETUP_INPUT_PREFIX"],
                    output_prefix=values["SETUP_OUTPUT_PREFIX"]
                )
                
                # Save config
                self.config.save_config(profile_file)
                self.window["STATUS"].update(f"Configuration saved as profile: {profile_name}")
                sg.popup("Profile saved", f"Your settings have been saved as profile: {profile_name}")
            except Exception as e:
                self.window["STATUS"].update(f"Error saving profile: {e}")
                sg.popup_error(f"Error saving profile: {e}")
                
        elif event == "SETUP_RESET":
            # Reset form to default values
            self.window["SETUP_MODEL"].update("gemini-1.5-flash-001")
            self.window["SETUP_EPOCHS"].update("5")
            self.window["SETUP_BATCH_SIZE"].update("4")
            self.window["SETUP_LEARNING_RATE"].update("1.0")
            self.window["SETUP_DELIMITER"].update("---")
            self.window["SETUP_INPUT_PREFIX"].update("INPUT:")
            self.window["SETUP_OUTPUT_PREFIX"].update("OUTPUT:")
            self.window["STATUS"].update("Form reset to default values")
    
    def handle_process_events(self, event: str, values: Dict[str, Any]) -> None:
        """
        Handle process files screen events.
        
        Args:
            event: Event name
            values: Form values
        """
        if event == "PROCESS_ADD_FILES":
            file_paths = sg.popup_get_file("Select input files", multiple_files=True)
            if file_paths:
                # Split the string into a list of file paths
                file_paths = file_paths.split(";")
                # Add to listbox
                current_files = list(self.window["PROCESS_FILES"].get_list_values())
                current_files.extend(file_paths)
                self.window["PROCESS_FILES"].update(values=current_files)
                self.window["STATUS"].update(f"Added {len(file_paths)} file(s)")
                
        elif event == "PROCESS_ADD_DIR":
            folder = sg.popup_get_folder("Select folder with input files")
            if folder:
                # Get all text files in the folder
                file_paths = [os.path.join(folder, f) for f in os.listdir(folder) 
                             if os.path.isfile(os.path.join(folder, f)) and f.endswith('.txt')]
                # Add to listbox
                current_files = list(self.window["PROCESS_FILES"].get_list_values())
                current_files.extend(file_paths)
                self.window["PROCESS_FILES"].update(values=current_files)
                self.window["STATUS"].update(f"Added {len(file_paths)} file(s) from folder")
                
        elif event == "PROCESS_REMOVE_FILE":
            selected_indices = self.window["PROCESS_FILES"].get_indexes()
            if selected_indices:
                current_files = list(self.window["PROCESS_FILES"].get_list_values())
                # Remove selected files
                new_files = [f for i, f in enumerate(current_files) if i not in selected_indices]
                self.window["PROCESS_FILES"].update(values=new_files)
                self.window["STATUS"].update(f"Removed {len(selected_indices)} file(s)")
                
        elif event == "PROCESS_BROWSE_OUTPUT":
            # This is handled by PySimpleGUI's FileSaveAs element
            pass
            
        elif event == "PROCESS_START":
            file_paths = list(self.window["PROCESS_FILES"].get_list_values())
            if not file_paths:
                sg.popup_error("No input files selected", "Please add at least one input file.")
                return
                
            output_file = values["PROCESS_OUTPUT_FILE"]
            if not output_file:
                sg.popup_error("No output file specified", "Please specify an output file.")
                return
                
            # Get format settings
            delimiter = values["PROCESS_DELIMITER"]
            input_prefix = values["PROCESS_INPUT_PREFIX"]
            output_prefix = values["PROCESS_OUTPUT_PREFIX"]
            
            try:
                # Process files
                self.window["STATUS"].update("Processing files...")
                
                if len(file_paths) == 1:
                    self.training_data = self.processor.process_file(
                        file_paths[0],
                        delimiter=delimiter,
                        input_prefix=input_prefix,
                        output_prefix=output_prefix
                    )
                else:
                    self.training_data = self.processor.process_multiple_files(
                        file_paths,
                        delimiter=delimiter,
                        input_prefix=input_prefix,
                        output_prefix=output_prefix
                    )
                
                # Preview if requested
                if values["PROCESS_PREVIEW"]:
                    self.preview_training_data()
                
                # Save data
                self.processor.save_training_data(self.training_data, output_file)
                self.window["STATUS"].update(f"Processed {len(self.training_data)} examples and saved to {output_file}")
                
                # Auto-train if requested
                if values["PROCESS_AUTO_TRAIN"]:
                    # Switch to train tab
                    self.handle_navigation("NAV_TRAIN")
                    # Update data source
                    self.window["TRAIN_DATA_SOURCE"].update(output_file)
                    # Update stats
                    self.window["TRAIN_DATA_STATS"].update(f"Examples: {len(self.training_data)}  Valid: {len(self.training_data)}  Invalid: 0")
                else:
                    sg.popup("Processing complete", f"Processed {len(self.training_data)} examples and saved to {output_file}")
            except Exception as e:
                self.window["STATUS"].update(f"Error processing files: {e}")
                sg.popup_error(f"Error processing files: {e}")
                
        elif event == "PROCESS_CANCEL":
            self.window["PROCESS_FILES"].update(values=[])
            self.window["STATUS"].update("Process canceled")
    
    def handle_train_events(self, event: str, values: Dict[str, Any]) -> None:
        """
        Handle train model screen events.
        
        Args:
            event: Event name
            values: Form values
        """
        if event == "TRAIN_BROWSE_DATA":
            # This is handled by PySimpleGUI's FileBrowse element
            pass
            
        elif event == "TRAIN_VIEW_EXAMPLES":
            data_source = values["TRAIN_DATA_SOURCE"]
            if not data_source or not os.path.isfile(data_source):
                sg.popup_error("Invalid data source", "Please select a valid training data file.")
                return
                
            try:
                # Load training data
                self.training_data = self.processor.load_training_data(data_source)
                # Update stats
                self.window["TRAIN_DATA_STATS"].update(f"Examples: {len(self.training_data)}  Valid: {len(self.training_data)}  Invalid: 0")
                # Preview data
                self.preview_training_data()
            except Exception as e:
                self.window["STATUS"].update(f"Error loading training data: {e}")
                sg.popup_error(f"Error loading training data: {e}")
                
        elif event == "TRAIN_EDIT_EXAMPLES":
            sg.popup_notify("This feature is not yet implemented", "The ability to edit examples will be added in a future version.")
            
        elif event == "TRAIN_START":
            data_source = values["TRAIN_DATA_SOURCE"]
            if not data_source or not os.path.isfile(data_source):
                sg.popup_error("Invalid data source", "Please select a valid training data file.")
                return
                
            # Check if API key is available
            if not self.config.get('api_key'):
                sg.popup_error("No API key", "Please set your Google AI API key in the Setup tab.")
                return
                
            try:
                # Load training data if not already loaded
                if not self.training_data:
                    self.training_data = self.processor.load_training_data(data_source)
                    # Update stats
                    self.window["TRAIN_DATA_STATS"].update(f"Examples: {len(self.training_data)}  Valid: {len(self.training_data)}  Invalid: 0")
                
                # Get training parameters
                model_name = values["TRAIN_MODEL"]
                display_name = values["TRAIN_DISPLAY_NAME"]
                epochs = int(values["TRAIN_EPOCHS"])
                batch_size = int(values["TRAIN_BATCH_SIZE"])
                learning_rate = float(values["TRAIN_LEARNING_RATE"])
                
                # Start training in a separate thread to avoid freezing the UI
                import threading
                
                def train_thread():
                    try:
                        self.window["STATUS"].update("Training model... This may take a while.")
                        self.window["TRAIN_START"].update(disabled=True)
                        
                        model_id = self.model_manager.train_model(
                            training_data=self.training_data,
                            model_name=model_name,
                            display_name=display_name,
                            epochs=epochs,
                            batch_size=batch_size,
                            learning_rate_multiplier=learning_rate
                        )
                        
                        if model_id:
                            # Update UI from the main thread
                            sg.popup("Training complete", f"Model training completed successfully!\nModel ID: {model_id}")
                            self.window["STATUS"].update(f"Model training completed. Model ID: {model_id}")
                        else:
                            sg.popup_error("Training failed", "Model training failed. Check the logs for details.")
                            self.window["STATUS"].update("Model training failed.")
                    except Exception as e:
                        sg.popup_error(f"Error training model: {e}")
                        self.window["STATUS"].update(f"Error training model: {e}")
                    finally:
                        self.window["TRAIN_START"].update(disabled=False)
                
                # Start the training thread
                threading.Thread(target=train_thread, daemon=True).start()
                
            except Exception as e:
                self.window["STATUS"].update(f"Error starting training: {e}")
                sg.popup_error(f"Error starting training: {e}")
                
        elif event == "TRAIN_SAVE_SETTINGS":
            try:
                # Update config with form values
                self.config.update(
                    model_name=values["TRAIN_MODEL"],
                    epochs=int(values["TRAIN_EPOCHS"]),
                    batch_size=int(values["TRAIN_BATCH_SIZE"]),
                    learning_rate=float(values["TRAIN_LEARNING_RATE"])
                )
                
                # Save config
                self.config.save_config()
                self.window["STATUS"].update("Training settings saved as default")
                sg.popup("Settings saved", "Your training settings have been saved as the default configuration.")
            except Exception as e:
                self.window["STATUS"].update(f"Error saving settings: {e}")
                sg.popup_error(f"Error saving settings: {e}")
                
        elif event == "TRAIN_CANCEL":
            # Reset form
            self.window["TRAIN_DATA_SOURCE"].update("")
            self.window["TRAIN_DATA_STATS"].update("Examples: 0  Valid: 0  Invalid: 0")
            self.window["TRAIN_DISPLAY_NAME"].update(f"Tuned Model - {datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.training_data = []
            self.window["STATUS"].update("Training canceled")
    
    def handle_test_events(self, event: str, values: Dict[str, Any]) -> None:
        """
        Handle test model screen events.
        
        Args:
            event: Event name
            values: Form values
        """
        if event == "TEST_REFRESH_MODELS":
            self.refresh_model_list()
            
        elif event == "TEST_LOAD_INPUT":
            file_path = sg.popup_get_file("Select input file", file_types=(("Text Files", "*.txt"),))
            if file_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.window["TEST_INPUT"].update(f.read())
                    self.window["STATUS"].update(f"Loaded test input from {file_path}")
                except Exception as e:
                    self.window["STATUS"].update(f"Error loading input file: {e}")
                    sg.popup_error(f"Error loading input file: {e}")
                    
        elif event == "TEST_GENERATE":
            model_id = values["TEST_MODEL"]
            test_input = values["TEST_INPUT"]
            
            if not model_id:
                sg.popup_error("No model selected", "Please select a model to test.")
                return
                
            if not test_input:
                sg.popup_error("No input provided", "Please enter some text to test the model with.")
                return
                
            # Check if API key is available
            if not self.config.get('api_key'):
                sg.popup_error("No API key", "Please set your Google AI API key in the Setup tab.")
                return
                
            try:
                self.window["STATUS"].update("Generating response...")
                self.window["TEST_GENERATE"].update(disabled=True)
                
                # Start generation in a separate thread to avoid freezing the UI
                import threading
                
                def generate_thread():
                    try:
                        response = self.model_manager.test_model(model_id, test_input)
                        
                        if response:
                            # Update UI from the main thread
                            self.window["TEST_RESPONSE"].update(response)
                            self.window["STATUS"].update("Response generated successfully")
                        else:
                            sg.popup_error("Generation failed", "Failed to generate a response. Check the logs for details.")
                            self.window["STATUS"].update("Response generation failed.")
                    except Exception as e:
                        sg.popup_error(f"Error generating response: {e}")
                        self.window["STATUS"].update(f"Error generating response: {e}")
                    finally:
                        self.window["TEST_GENERATE"].update(disabled=False)
                
                # Start the generation thread
                threading.Thread(target=generate_thread, daemon=True).start()
                
            except Exception as e:
                self.window["STATUS"].update(f"Error generating response: {e}")
                sg.popup_error(f"Error generating response: {e}")
                self.window["TEST_GENERATE"].update(disabled=False)
                
        elif event == "TEST_SAVE_RESPONSE":
            response = values["TEST_RESPONSE"]
            if not response:
                sg.popup_error("No response to save", "Generate a response first.")
                return
                
            file_path = sg.popup_get_file("Save response to file", save_as=True, file_types=(("Text Files", "*.txt"),))
            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(response)
                    self.window["STATUS"].update(f"Response saved to {file_path}")
                    sg.popup("Response saved", f"Response saved to {file_path}")
                except Exception as e:
                    self.window["STATUS"].update(f"Error saving response: {e}")
                    sg.popup_error(f"Error saving response: {e}")
                    
        elif event == "TEST_CLEAR":
            self.window["TEST_INPUT"].update("")
            self.window["TEST_RESPONSE"].update("")
            self.window["STATUS"].update("Test form cleared")
    
    def handle_model_events(self, event: str, values: Dict[str, Any]) -> None:
        """
        Handle model manager screen events.
        
        Args:
            event: Event name
            values: Form values
        """
        if event == "MODEL_REFRESH":
            self.refresh_model_list()
            
        elif event == "MODEL_TABLE":
            # Get selected model
            selected_row = values["MODEL_TABLE"]
            if selected_row:
                row_index = selected_row[0]
                model_info = self.model_list[row_index]
                
                # Update model details
                self.window["MODEL_NAME"].update(f"Name: {model_info['name']}")
                self.window["MODEL_ID"].update(f"ID: {model_info['id']}")
                self.window["MODEL_BASE"].update(f"Base Model: {model_info['base_model']}")
                self.window["MODEL_CREATED"].update(f"Created: {model_info['created']}")
                self.window["MODEL_STATE"].update(f"State: {model_info['state']}")
                
        elif event == "MODEL_DELETE":
            selected_row = values["MODEL_TABLE"]
            if not selected_row:
                sg.popup_error("No model selected", "Please select a model to delete.")
                return
                
            row_index = selected_row[0]
            model_info = self.model_list[row_index]
            
            # Confirm deletion
            if sg.popup_yes_no(f"Delete model '{model_info['name']}'?", "This action cannot be undone.") == "Yes":
                try:
                    self.window["STATUS"].update(f"Deleting model {model_info['name']}...")
                    success = self.model_manager.delete_model(model_info['id'])
                    
                    if success:
                        self.window["STATUS"].update(f"Model {model_info['name']} deleted successfully")
                        sg.popup("Model deleted", f"Model {model_info['name']} deleted successfully")
                        # Refresh model list
                        self.refresh_model_list()
                    else:
                        self.window["STATUS"].update(f"Failed to delete model {model_info['name']}")
                        sg.popup_error(f"Failed to delete model {model_info['name']}")
                except Exception as e:
                    self.window["STATUS"].update(f"Error deleting model: {e}")
                    sg.popup_error(f"Error deleting model: {e}")
                    
        elif event == "MODEL_TEST":
            selected_row = values["MODEL_TABLE"]
            if not selected_row:
                sg.popup_error("No model selected", "Please select a model to test.")
                return
                
            row_index = selected_row[0]
            model_info = self.model_list[row_index]
            
            # Switch to test tab
            self.handle_navigation("NAV_TEST")
            
            # Update model selection
            model_names = [m["name"] for m in self.model_list]
            self.window["TEST_MODEL"].update(value=model_info["name"], values=model_names)
            self.window["TEST_BASE_MODEL"].update(f"Base Model: {model_info['base_model']}")
            self.window["TEST_CREATED_DATE"].update(f"Created: {model_info['created']}")
            
        elif event == "MODEL_EXPORT":
            selected_row = values["MODEL_TABLE"]
            if not selected_row:
                sg.popup_error("No model selected", "Please select a model to export details.")
                return
                
            row_index = selected_row[0]
            model_info = self.model_list[row_index]
            
            file_path = sg.popup_get_file("Export model details", save_as=True, file_types=(("JSON Files", "*.json"),))
            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(model_info, f, indent=2)
                    self.window["STATUS"].update(f"Model details exported to {file_path}")
                    sg.popup("Export complete", f"Model details exported to {file_path}")
                except Exception as e:
                    self.window["STATUS"].update(f"Error exporting model details: {e}")
                    sg.popup_error(f"Error exporting model details: {e}")
                    
        elif event == "MODEL_CLONE":
            selected_row = values["MODEL_TABLE"]
            if not selected_row:
                sg.popup_error("No model selected", "Please select a model to clone settings.")
                return
                
            row_index = selected_row[0]
            model_info = self.model_list[row_index]
            
            # Switch to train tab
            self.handle_navigation("NAV_TRAIN")
            
            # Update training settings
            self.window["TRAIN_MODEL"].update(model_info["base_model"])
            self.window["TRAIN_DISPLAY_NAME"].update(f"Clone of {model_info['name']}")
            
            sg.popup("Settings cloned", f"Training settings cloned from model '{model_info['name']}'")
    
    def validate_api_key_status(self) -> None:
        """Validate API key and update status."""
        api_key = self.config.get('api_key')
        if api_key:
            if self.model_manager.validate_api_key(api_key):
                self.window["SETUP_API_STATUS"].update("API Status: Valid", text_color="green")
            else:
                self.window["SETUP_API_STATUS"].update("API Status: Invalid", text_color="red")
        else:
            self.window["SETUP_API_STATUS"].update("API Status: No key provided", text_color="orange")
    
    def refresh_model_list(self) -> None:
        """Refresh the list of tuned models."""
        self.window["STATUS"].update("Refreshing model list...")
        
        try:
            # Get model list
            self.model_list = self.model_manager.list_models()
            
            if self.model_list:
                # Update model table
                table_data = [[m["name"], m["created"], m["base_model"], m["state"]] for m in self.model_list]
                self.window["MODEL_TABLE"].update(values=table_data)
                
                # Update model dropdown in test tab
                model_names = [m["name"] for m in self.model_list]
                self.window["TEST_MODEL"].update(values=model_names)
                
                self.window["STATUS"].update(f"Found {len(self.model_list)} tuned models")
            else:
                self.window["MODEL_TABLE"].update(values=[])
                self.window["TEST_MODEL"].update(values=[])
                self.window["STATUS"].update("No tuned models found")
        except Exception as e:
            self.window["STATUS"].update(f"Error refreshing model list: {e}")
            sg.popup_error(f"Error refreshing model list: {e}")
    
    def preview_training_data(self) -> None:
        """Preview training data examples."""
        if not self.training_data:
            sg.popup_error("No training data", "No training data available to preview.")
            return
            
        # Create preview window
        preview_layout = [
            [sg.Text("Training Data Preview", font=("Helvetica", 16))],
            [sg.Table(
                values=[[i+1, ex["text_input"][:50] + "..." if len(ex["text_input"]) > 50 else ex["text_input"], 
                        ex["output"][:50] + "..." if len(ex["output"]) > 50 else ex["output"]] 
                       for i, ex in enumerate(self.training_data[:10])],
                headings=["#", "Input", "Output"],
                auto_size_columns=False,
                col_widths=[5, 40, 40],
                justification="left",
                num_rows=min(10, len(self.training_data))
            )],
            [sg.Text(f"Showing {min(10, len(self.training_data))} of {len(self.training_data)} examples")],
            [sg.Button("Close")]
        ]
        
        preview_window = sg.Window("Training Data Preview", preview_layout, modal=True)
        
        while True:
            event, values = preview_window.read()
            if event == sg.WINDOW_CLOSED or event == "Close":
                break
                
        preview_window.close()


def check_requirements() -> List[str]:
    """
    Check for missing required packages.
    
    Returns:
        List of missing package names
    """
    missing_packages = []
    
    if not HAS_GENAI:
        missing_packages.append("google-generativeai")
    
    if not HAS_DOTENV:
        missing_packages.append("python-dotenv")
    
    if not HAS_TQDM:
        missing_packages.append("tqdm")
        
    return missing_packages


def install_packages(packages: List[str]) -> bool:
    """
    Install missing packages.
    
    Args:
        packages: List of package names to install
        
    Returns:
        True if installation was successful, False otherwise
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        logger.info(f"Successfully installed: {', '.join(packages)}")
        return True
    except Exception as e:
        logger.error(f"Error installing packages: {e}")
        return False


def main() -> None:
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(description="Enhanced Gemini Tuner")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--gui", action="store_true", help="Run in graphical user interface mode")
    mode_group.add_argument("--interactive", action="store_true", help="Run in interactive command-line mode")
    
    # Common options
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", action="store_true", help="Suppress all output except errors")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    
    # Command-specific options
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process input files into training data")
    process_parser.add_argument("--input", type=str, nargs="+", required=True, help="Input file path(s)")
    process_parser.add_argument("--output", type=str, default="training_data.json", help="Output file path")
    process_parser.add_argument("--delimiter", type=str, help="Example delimiter")
    process_parser.add_argument("--input-prefix", type=str, help="Input prefix")
    process_parser.add_argument("--output-prefix", type=str, help="Output prefix")
    process_parser.add_argument("--preview", action="store_true", help="Preview processed data")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model using processed data")
    train_parser.add_argument("--data", type=str, required=True, help="Training data file path")
    train_parser.add_argument("--model", type=str, help="Base model name")
    train_parser.add_argument("--name", type=str, help="Display name for the tuned model")
    train_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, help="Batch size for training")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate multiplier")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test a tuned model")
    test_parser.add_argument("--model-id", type=str, required=True, help="Tuned model ID")
    test_parser.add_argument("--input", type=str, required=True, help="Test input text or file path")
    test_parser.add_argument("--output", type=str, help="Output file path for response")
    
    # Models command
    models_parser = subparsers.add_parser("models", help="List, describe or delete tuned models")
    models_parser.add_argument("--list", action="store_true", help="List all tuned models")
    models_parser.add_argument("--delete", type=str, help="Delete a tuned model by ID")
    models_parser.add_argument("--export", type=str, help="Export model details to file")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Create configuration files and validate setup")
    setup_parser.add_argument("--create-config", action="store_true", help="Create a sample configuration file")
    setup_parser.add_argument("--validate-api", action="store_true", help="Validate API key")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show version and exit
    if args.version:
        print("Enhanced Gemini Tuner v1.0.0")
        return
    
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    
    # Check requirements
    missing_packages = check_requirements()
    if missing_packages:
        logger.warning(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Installing required packages...")
        
        if not install_packages(missing_packages):
            logger.error("Failed to install required packages. Please install them manually:")
            logger.error(f"pip install {' '.join(missing_packages)}")
            
            # Continue with limited functionality
            logger.warning("Continuing with limited functionality.")
        else:
            # Reload modules
            if "google-generativeai" in missing_packages:
                import google.generativeai as genai
                globals()['HAS_GENAI'] = True
                globals()['genai'] = genai
            
            if "python-dotenv" in missing_packages:
                from dotenv import load_dotenv
                globals()['HAS_DOTENV'] = True
                globals()['load_dotenv'] = load_dotenv
            
            if "tqdm" in missing_packages:
                from tqdm import tqdm
                globals()['HAS_TQDM'] = True
                globals()['tqdm'] = tqdm
                
            if "PySimpleGUI" in missing_packages:
                try:
                    import PySimpleGUI as sg
                    globals()['HAS_GUI'] = True
                    globals()['sg'] = sg
                except ImportError:
                    pass
    
    # Initialize configuration
    config = Config(args.config)
    
    # Initialize components
    processor = DataProcessor(config)
    model_manager = ModelManager(config)
    
    # Run in GUI mode
    if args.gui:
        if not HAS_GUI:
            logger.error("PySimpleGUI not installed. Cannot run in GUI mode.")
            logger.info("Installing PySimpleGUI...")
            
            if install_packages(["PySimpleGUI"]):
                import PySimpleGUI as sg
                globals()['HAS_GUI'] = True
                globals()['sg'] = sg
            else:
                logger.error("Failed to install PySimpleGUI. Falling back to interactive mode.")
                interactive_cli = InteractiveCLI(config, processor, model_manager)
                interactive_cli.run()
                return
        
        gui = GUI(config, processor, model_manager)
        gui.run()
        return
    
    # Run in interactive mode
    if args.interactive:
        interactive_cli = InteractiveCLI(config, processor, model_manager)
        interactive_cli.run()
        return
    
    # Handle commands
    if args.command == "process":
        # Process input files
        try:
            if len(args.input) == 1:
                training_data = processor.process_file(
                    args.input[0],
                    delimiter=args.delimiter,
                    input_prefix=args.input_prefix,
                    output_prefix=args.output_prefix
                )
            else:
                training_data = processor.process_multiple_files(
                    args.input,
                    delimiter=args.delimiter,
                    input_prefix=args.input_prefix,
                    output_prefix=args.output_prefix
                )
            
            # Preview if requested
            if args.preview:
                processor.preview_training_data(training_data)
            
            # Save data
            processor.save_training_data(training_data, args.output)
        except Exception as e:
            logger.error(f"Error processing files: {e}")
            return
    
    elif args.command == "train":
        # Train a model
        try:
            # Load training data
            training_data = processor.load_training_data(args.data)
            
            # Train model
            model_id = model_manager.train_model(
                training_data=training_data,
                model_name=args.model,
                display_name=args.name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate_multiplier=args.learning_rate
            )
            
            if model_id:
                logger.info(f"Model training completed successfully! Model ID: {model_id}")
            else:
                logger.error("Model training failed.")
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return
    
    elif args.command == "test":
        # Test a model
        try:
            # Check if input is a file
            if os.path.isfile(args.input):
                with open(args.input, 'r', encoding='utf-8') as f:
                    test_input = f.read()
            else:
                test_input = args.input
            
            # Test model
            response = model_manager.test_model(args.model_id, test_input)
            
            if response:
                if args.output:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        f.write(response)
                    logger.info(f"Response saved to {args.output}")
                else:
                    print("\nModel response:")
                    print("==============")
                    print(response)
            else:
                logger.error("Failed to generate a response.")
        except Exception as e:
            logger.error(f"Error testing model: {e}")
            return
    
    elif args.command == "models":
        # List models
        if args.list:
            try:
                models = model_manager.list_models()
                
                if models:
                    print("\nTuned Models:")
                    print("============")
                    
                    for i, model in enumerate(models):
                        print(f"{i+1}. {model['name']}")
                        print(f"   ID: {model['id']}")
                        print(f"   Base Model: {model['base_model']}")
                        print(f"   Created: {model['created']}")
                        print(f"   State: {model['state']}")
                        print()
                else:
                    logger.info("No tuned models found.")
            except Exception as e:
                logger.error(f"Error listing models: {e}")
                return
        
        # Delete model
        if args.delete:
            try:
                success = model_manager.delete_model(args.delete)
                
                if success:
                    logger.info(f"Model {args.delete} deleted successfully.")
                else:
                    logger.error(f"Failed to delete model {args.delete}.")
            except Exception as e:
                logger.error(f"Error deleting model: {e}")
                return
        
        # Export model details
        if args.export:
            try:
                models = model_manager.list_models()
                
                if models:
                    with open(args.export, 'w', encoding='utf-8') as f:
                        json.dump(models, f, indent=2)
                    logger.info(f"Model details exported to {args.export}")
                else:
                    logger.error("No models found to export.")
            except Exception as e:
                logger.error(f"Error exporting model details: {e}")
                return
    
    elif args.command == "setup":
        # Create configuration file
        if args.create_config:
            try:
                Config.create_sample_config()
            except Exception as e:
                logger.error(f"Error creating configuration file: {e}")
                return
        
        # Validate API key
        if args.validate_api:
            api_key = config.get('api_key')
            
            if not api_key:
                logger.error("No API key found in configuration.")
                return
            
            if model_manager.validate_api_key(api_key):
                logger.info("API key is valid.")
            else:
                logger.error("API key is invalid.")
    
    else:
        # No command specified, show help
        parser.print_help()


if __name__ == "__main__":
    main()
