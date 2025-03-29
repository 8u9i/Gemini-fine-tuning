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
    - google-genai (Google AI Python SDK)
    - python-dotenv (for .env file support)
    - tqdm (for progress bars)
    - PySimpleGUI (for graphical interface)
    - Valid Google AI API key with tuning permissions
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
import threading
import getpass # Used in InteractiveCLI

# --- Dependency Checks and Imports ---

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('GeminiTuner')

# Attempt to import required libraries, track availability
try:
    from google import genai  # New import structure
    from google.genai import types  # Updated types import
    from google.api_core import exceptions as google_exceptions
    HAS_GENAI = True
except ImportError:
    genai = None
    types = None
    google_exceptions = None
    HAS_GENAI = False

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    load_dotenv = None
    HAS_DOTENV = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    tqdm = None
    HAS_TQDM = False

try:
    import PySimpleGUI as sg
    HAS_GUI = True
except ImportError:
    sg = None
    HAS_GUI = False

# --- Configuration Class ---

class Config:
    """Configuration management for Gemini Tuner."""

    DEFAULT_CONFIG = {
        "api_key": "",
        "base_model": "models/gemini-1.0-pro-001", # Tunable base model
        "epochs": 5,
        "batch_size": 4,
        "learning_rate_multiplier": 1.0,
        "delimiter": "---",
        "input_prefix": "INPUT:",
        "output_prefix": "OUTPUT:",
    }

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy() # Start with defaults
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from environment variables and .env file."""
        env_file_loaded = False
        if HAS_DOTENV:
            env_path = self.config_file or self._find_env_file()
            if env_path and os.path.isfile(env_path):
                try:
                    load_dotenv(dotenv_path=env_path, override=True)
                    self.config_file = env_path
                    logger.info(f"Loaded configuration from {env_path}")
                    env_file_loaded = True
                except Exception as e:
                    logger.warning(f"Could not load .env file at {env_path}: {e}")

        # Load from environment variables (these override .env and defaults)
        self.config["api_key"] = os.environ.get("GOOGLE_API_KEY", self.config["api_key"])
        self.config["base_model"] = os.environ.get("GEMINI_BASE_MODEL", self.config["base_model"])
        self.config["epochs"] = int(os.environ.get("GEMINI_EPOCHS", self.config["epochs"]))
        self.config["batch_size"] = int(os.environ.get("GEMINI_BATCH_SIZE", self.config["batch_size"]))
        self.config["learning_rate_multiplier"] = float(os.environ.get("GEMINI_LEARNING_RATE_MULTIPLIER", self.config["learning_rate_multiplier"]))
        self.config["delimiter"] = os.environ.get("EXAMPLE_DELIMITER", self.config["delimiter"])
        self.config["input_prefix"] = os.environ.get("INPUT_PREFIX", self.config["input_prefix"])
        self.config["output_prefix"] = os.environ.get("OUTPUT_PREFIX", self.config["output_prefix"])

        if not env_file_loaded and not self.config.get("api_key"):
             logger.debug("No .env file found or loaded, and GOOGLE_API_KEY not set.")


    def save_config(self, file_path: Optional[str] = None) -> None:
        """Save configuration to a file."""
        save_path = file_path or self.config_file or ".env"
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(f"# Gemini Tuner Configuration - Saved {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"# Google AI API key (required for model tuning)\n")
                f.write(f"GOOGLE_API_KEY={self.config.get('api_key', '')}\n\n")
                f.write(f"# Default base model for tuning (must be a tunable model like models/gemini-1.0-pro-001)\n")
                f.write(f"GEMINI_BASE_MODEL={self.config.get('base_model')}\n\n")
                f.write(f"# Default settings for tuning\n")
                f.write(f"GEMINI_EPOCHS={self.config.get('epochs')}\n")
                f.write(f"GEMINI_BATCH_SIZE={self.config.get('batch_size')}\n")
                f.write(f"GEMINI_LEARNING_RATE_MULTIPLIER={self.config.get('learning_rate_multiplier')}\n\n")
                f.write(f"# Default file format settings\n")
                f.write(f"EXAMPLE_DELIMITER={self.config.get('delimiter')}\n")
                f.write(f"INPUT_PREFIX={self.config.get('input_prefix')}\n")
                f.write(f"OUTPUT_PREFIX={self.config.get('output_prefix')}\n")
            logger.info(f"Configuration saved to {save_path}")
            self.config_file = save_path
        except IOError as e:
            logger.error(f"Failed to save configuration to {save_path}: {e}")

    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if key in self.config:
                # Basic type conversion for known numeric fields
                if key in ['epochs', 'batch_size'] and value is not None:
                    try:
                        self.config[key] = int(value)
                    except (ValueError, TypeError):
                         logger.warning(f"Invalid value for {key}: {value}. Keeping previous value: {self.config[key]}")
                elif key == 'learning_rate_multiplier' and value is not None:
                     try:
                        self.config[key] = float(value)
                     except (ValueError, TypeError):
                         logger.warning(f"Invalid value for {key}: {value}. Keeping previous value: {self.config[key]}")
                else:
                    self.config[key] = value
            else:
                logger.warning(f"Attempted to update unknown config key: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def _find_env_file(self, max_depth: int = 5) -> Optional[str]:
        """Find .env file in current directory or parent directories."""
        current_dir = Path.cwd()
        for _ in range(max_depth + 1):
            env_file = current_dir / '.env'
            if env_file.is_file():
                return str(env_file)
            if current_dir.parent == current_dir: # Reached root
                break
            current_dir = current_dir.parent
        return None

    @staticmethod
    def create_sample_config(file_path: str = '.env.example') -> None:
        """Create a sample configuration file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"""# Gemini Tuner Configuration Example - Saved {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# --- REQUIRED ---
# Google AI API key (obtain from Google AI Studio: https://aistudio.google.com/app/apikey)
GOOGLE_API_KEY=YOUR_API_KEY_HERE

# --- TUNING DEFAULTS (Optional) ---
# Base model for tuning (must be a tunable model, check Google AI documentation for current list)
# Example: models/gemini-1.0-pro-001
GEMINI_BASE_MODEL={Config.DEFAULT_CONFIG['base_model']}

# Number of training epochs
GEMINI_EPOCHS={Config.DEFAULT_CONFIG['epochs']}

# Batch size during training
GEMINI_BATCH_SIZE={Config.DEFAULT_CONFIG['batch_size']}

# Learning rate multiplier (adjusts the base learning rate)
GEMINI_LEARNING_RATE_MULTIPLIER={Config.DEFAULT_CONFIG['learning_rate_multiplier']}

# --- FILE PROCESSING DEFAULTS (Optional) ---
# Delimiter separating examples in your input text file
EXAMPLE_DELIMITER="{Config.DEFAULT_CONFIG['delimiter']}"

# Prefix indicating the start of an input part within an example
INPUT_PREFIX="{Config.DEFAULT_CONFIG['input_prefix']}"

# Prefix indicating the start of the expected output part within an example
OUTPUT_PREFIX="{Config.DEFAULT_CONFIG['output_prefix']}"
""")
            logger.info(f"Sample configuration file created at {file_path}")
            logger.info(f"ACTION REQUIRED: Copy this file to '.env' in your project root and replace 'YOUR_API_KEY_HERE' with your actual Google AI API key.")
        except IOError as e:
             logger.error(f"Failed to create sample configuration file at {file_path}: {e}")

# --- Data Processor Class ---

class DataProcessor:
    """Process text files into training data for Gemini model tuning."""

    # Default limits (can be overridden, check latest Gemini docs)
    DEFAULT_MAX_INPUT_CHARS = 10000 # Conservative default
    DEFAULT_MAX_OUTPUT_CHARS = 2000 # Conservative default

    def __init__(self, config: Config):
        """Initialize data processor."""
        self.config = config
        # Fetch limits from Gemini model info if possible, otherwise use defaults
        # This requires an API call, maybe do it lazily or in ModelManager
        self.max_input_chars = self.DEFAULT_MAX_INPUT_CHARS
        self.max_output_chars = self.DEFAULT_MAX_OUTPUT_CHARS


    def process_file(self,
                    file_path: str,
                    delimiter: Optional[str] = None,
                    input_prefix: Optional[str] = None,
                    output_prefix: Optional[str] = None
                    ) -> Tuple[List[Dict[str, str]], int, int]:
        """
        Process a text file into a format suitable for Gemini model tuning.

        Args:
            file_path: Path to the text file.
            delimiter: String that separates examples in the file.
            input_prefix: Prefix that marks the beginning of input text.
            output_prefix: Prefix that marks the beginning of output text.

        Returns:
            A tuple containing:
            - List of valid examples (dictionaries with 'text_input' and 'output').
            - Count of valid examples.
            - Count of invalid/skipped examples.
        """
        delimiter = delimiter or self.config.get('delimiter')
        input_prefix = input_prefix or self.config.get('input_prefix')
        output_prefix = output_prefix or self.config.get('output_prefix')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise IOError(f"Error reading file {file_path}: {e}")

        examples_raw = content.split(delimiter)
        training_data = []
        valid_count = 0
        invalid_count = 0

        example_iterator = tqdm(enumerate(examples_raw), total=len(examples_raw), desc=f"Processing {Path(file_path).name}", leave=False) if HAS_TQDM else enumerate(examples_raw)

        for i, example_raw in example_iterator:
            example = example_raw.strip()
            if not example:
                continue

            input_start_idx = example.find(input_prefix)
            output_start_idx = example.find(output_prefix)

            # Basic validation
            if input_start_idx == -1:
                logger.warning(f"Example {i+1} in {file_path}: Missing input prefix '{input_prefix}'. Skipping.")
                invalid_count += 1
                continue
            if output_start_idx == -1:
                logger.warning(f"Example {i+1} in {file_path}: Missing output prefix '{output_prefix}'. Skipping.")
                invalid_count += 1
                continue
            if output_start_idx < input_start_idx + len(input_prefix):
                 logger.warning(f"Example {i+1} in {file_path}: Output prefix appears before or inside input text. Skipping.")
                 invalid_count += 1
                 continue

            # Extract text
            input_text = example[input_start_idx + len(input_prefix):output_start_idx].strip()
            output_text = example[output_start_idx + len(output_prefix):].strip()

            if not input_text:
                 logger.warning(f"Example {i+1} in {file_path}: Input text is empty after stripping. Skipping.")
                 invalid_count += 1
                 continue
            if not output_text:
                 logger.warning(f"Example {i+1} in {file_path}: Output text is empty after stripping. Skipping.")
                 invalid_count += 1
                 continue

            # Validate lengths (using current limits)
            input_len = len(input_text)
            output_len = len(output_text)
            truncated = False
            if input_len > self.max_input_chars:
                logger.warning(f"Example {i+1} in {file_path}: Input ({input_len} chars) exceeds limit ({self.max_input_chars}). Truncating.")
                input_text = input_text[:self.max_input_chars]
                truncated = True
            if output_len > self.max_output_chars:
                logger.warning(f"Example {i+1} in {file_path}: Output ({output_len} chars) exceeds limit ({self.max_output_chars}). Truncating.")
                output_text = output_text[:self.max_output_chars]
                truncated = True

            # Add valid (possibly truncated) example
            training_data.append({"text_input": input_text, "output": output_text})
            valid_count += 1
            if truncated:
                 # If truncation happened, it's still counted as valid for training, but maybe log differently?
                 pass

        if HAS_TQDM: example_iterator.close() # Close tqdm bar

        logger.info(f"Finished processing {file_path}: {valid_count} valid examples, {invalid_count} skipped.")
        return training_data, valid_count, invalid_count

    def process_multiple_files(self, file_paths: List[str], **kwargs) -> Tuple[List[Dict[str, str]], int, int]:
        """Process multiple text files into a single training dataset."""
        all_training_data = []
        total_valid = 0
        total_invalid = 0

        for file_path in file_paths:
            try:
                data, valid, invalid = self.process_file(file_path, **kwargs)
                all_training_data.extend(data)
                total_valid += valid
                total_invalid += invalid
            except FileNotFoundError:
                logger.error(f"Skipping file: {file_path} (Not found)")
                total_invalid += 1 # Count file not found as an issue
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}. Skipping file.")
                total_invalid += 1 # Count processing error as an issue

        logger.info(f"Finished processing all files. Total: {total_valid} valid examples, {total_invalid} skipped/errors.")
        # Check minimum dataset size (Gemini usually requires >= 10)
        if total_valid < 10:
             logger.warning(f"Dataset size ({total_valid}) is very small. Tuning may not be effective or might fail. Recommended size is typically 100+ examples.")
        elif total_valid < 100:
             logger.warning(f"Dataset size ({total_valid}) is small. Consider adding more examples for better tuning results.")

        return all_training_data, total_valid, total_invalid

    def save_training_data(self, training_data: List[Dict[str, str]], output_file: str) -> None:
        """Save the processed training data to a JSONL file (JSON Lines format is often preferred)."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for example in training_data:
                    f.write(json.dumps(example) + '\n')
            logger.info(f"Training data ({len(training_data)} examples) saved to {output_path} (JSONL format)")
        except IOError as e:
            logger.error(f"Error saving training data to {output_path}: {e}")
            raise # Re-raise the exception

    def load_training_data(self, input_file: str) -> List[Dict[str, str]]:
        """Load training data from a JSONL file."""
        input_path = Path(input_file)
        if not input_path.is_file():
             raise FileNotFoundError(f"Training data file not found: {input_path}")

        training_data = []
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        try:
                            example = json.loads(line)
                            if isinstance(example, dict) and 'text_input' in example and 'output' in example:
                                training_data.append(example)
                            else:
                                logger.warning(f"Skipping invalid line {i+1} in {input_path}: Missing keys or not a dict.")
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid JSON on line {i+1} in {input_path}.")
            logger.info(f"Loaded {len(training_data)} examples from {input_path}")
            return training_data
        except IOError as e:
            logger.error(f"Error loading training data from {input_path}: {e}")
            raise # Re-raise the exception

    def preview_training_data(self, training_data: List[Dict[str, str]], num_examples: int = 3) -> None:
        """Preview training data examples to the console."""
        if not training_data:
            print("\nNo training data to preview.")
            return

        print(f"\n--- Training Data Preview (First {min(num_examples, len(training_data))} of {len(training_data)}) ---")
        for i, example in enumerate(training_data[:num_examples]):
            in_preview = example.get('text_input', 'N/A')[:150] + ('...' if len(example.get('text_input', '')) > 150 else '')
            out_preview = example.get('output', 'N/A')[:150] + ('...' if len(example.get('output', '')) > 150 else '')
            print(f"\nExample {i+1}:")
            print(f"  Input:  \"{in_preview}\"")
            print(f"  Output: \"{out_preview}\"")
        print("--- End Preview ---")


# --- Chat Manager Class ---
class ChatManager:
    """Manages active chat sessions with Gemini models."""

    def __init__(self, model_manager: 'ModelManager'):
        self.model_manager = model_manager # Reference to access the client
        self.chats: Dict[str, Dict[str, Any]] = {} # Store chat sessions by ID
        self.next_chat_id = 1

    def create_chat(self, model_name: Optional[str] = None) -> Optional[str]:
        """Creates a new chat session."""
        if not self.model_manager.is_client_ready():
            logger.error("Cannot create chat: Google AI client is not initialized.")
            return None

        model_to_use = model_name or self.model_manager.config.get('base_model') # Default to base, could allow tuned
        try:
            # Ensure the model exists and is valid for generation
            # Note: get_model might raise if the model doesn't exist
            model_info = self.model_manager.client.get_model(model_to_use)
            if 'generateContent' not in model_info.supported_generation_methods:
                 logger.error(f"Model '{model_to_use}' does not support content generation.")
                 return None

            # Use the client's chats.create method directly
            chat_session = self.model_manager.client.chats.create(model=model_to_use, history=[])
            chat_id = f"chat_{self.next_chat_id}"
            self.next_chat_id += 1
            self.chats[chat_id] = {"session": chat_session, "model": model_to_use, "created": datetime.now()}
            logger.info(f"Created new chat session '{chat_id}' with model '{model_to_use}'.")
            return chat_id
        except Exception as e:
            logger.error(f"Failed to create chat session with model '{model_to_use}': {e}")
            return None

    def send_message(self, chat_id: str, message: str, stream: bool = False) -> Union[str, Generator[str, None, None], None]:
        """Sends a message to an existing chat session."""
        if chat_id not in self.chats:
            logger.error(f"Chat session '{chat_id}' not found.")
            return None
        if not self.model_manager.is_client_ready():
            logger.error("Cannot send message: Google AI client is not initialized.")
            return None

        chat_data = self.chats[chat_id]
        session = chat_data["session"]
        model_name = chat_data["model"]
        try:
            logger.debug(f"Sending message to chat '{chat_id}' (model: {model_name}): '{message[:50]}...'")
            response = session.send_message(message, stream=stream)

            if stream:
                def stream_generator():
                    try:
                        for chunk in response:
                            yield chunk.text
                    except Exception as e:
                         logger.error(f"Error during streaming response for chat '{chat_id}': {e}")
                         # You might want to yield an error message or handle differently
                return stream_generator()
            else:
                logger.debug(f"Received response for chat '{chat_id}': '{response.text[:50]}...'")
                return response.text
        except Exception as e:
            logger.error(f"Error sending message or receiving response for chat '{chat_id}': {e}")
            # Consider how to handle API errors (e.g., rate limits, blocked content)
            if hasattr(e, 'message'): # More specific error handling if available
                logger.error(f"API Error detail: {e.message}")
            return None

    def get_history(self, chat_id: str) -> List[Dict[str, str]]:
        """Gets the message history for a chat session."""
        if chat_id not in self.chats:
            logger.error(f"Chat session '{chat_id}' not found.")
            return []
        session = self.chats[chat_id]["session"]
        # Convert the internal history format to a simple list of dicts
        history = []
        if session.history:
            for content in session.history:
                 # Assuming Parts have text attribute
                 text_parts = [part.text for part in content.parts if hasattr(part, 'text')]
                 history.append({
                     "role": content.role,
                     "content": " ".join(text_parts) # Join parts if multiple exist
                 })
        return history


    def list_chats(self) -> List[Dict[str, Any]]:
        """Lists active chat sessions."""
        return [{"id": cid, "model": data["model"], "created": data["created"]}
                for cid, data in self.chats.items()]

    def delete_chat(self, chat_id: str) -> bool:
        """Deletes a chat session."""
        if chat_id in self.chats:
            del self.chats[chat_id]
            logger.info(f"Deleted chat session '{chat_id}'.")
            return True
        else:
            logger.warning(f"Attempted to delete non-existent chat session '{chat_id}'.")
            return False


# --- Model Manager Class ---

class ModelManager:
    """Manage Gemini model tuning, listing, deletion, testing, and chat."""

    TUNABLE_MODELS = [
        "models/gemini-1.0-pro-001-tuning",  # Tunable version of Gemini 1.0 Pro
        "models/gemini-1.5-flash-001-tuning"  # Tunable version of Gemini 1.5 Flash
    ] # Updated with correct tunable model names

    def __init__(self, config: Config):
        """Initialize model manager."""
        self.config = config
        self.client: Optional[genai.Client] = None
        self.chat_manager: Optional[ChatManager] = None
        self._initialize_genai()
        self.chat_manager = ChatManager(self) # Initialize ChatManager here

    def _initialize_genai(self) -> bool:
        """Initialize the Google Generative AI SDK client."""
        self.client = None # Reset client state
        if not HAS_GENAI:
            logger.error("Google Generative AI SDK (`google-genai`) not found. Install it to enable model operations.")
            return False

        api_key = self.config.get('api_key')
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found in config or environment. Model operations (tuning, testing, listing) disabled.")
            logger.warning("You can still use file processing features.")
            logger.info("To enable model features, set GOOGLE_API_KEY in your .env file or environment.")
            return False

        try:
            # Create a client instance with the API key
            self.client = genai.Client(api_key=api_key)

            # Test connection and API key validity by listing models
            logger.info("Connecting to Google AI API...")
            # Use the new SDK structure to list models
            available_models = list(self.client.models.list()) # Use list() to force iteration now
            logger.info(f"Successfully connected. Found {len(available_models)} available models.")

            # Optionally log tunable models found vs expected
            found_tunable = [m.name for m in available_models if m.name in self.TUNABLE_MODELS]
            logger.info(f"Confirmed tunable base models available: {found_tunable or 'None found matching known list'}")
            if not found_tunable:
                 logger.warning(f"Expected tunable models ({self.TUNABLE_MODELS}) not found in API list. Check base model names.")

            return True
        except google_exceptions.PermissionDenied as e:
             logger.error(f"API Key validation failed: Permission Denied. Check your API key and project permissions. Detail: {e}")
             self.client = None
             return False
        except google_exceptions.Unauthenticated as e:
             logger.error(f"API Key validation failed: Authentication Error. Is the key correct? Detail: {e}")
             self.client = None
             return False
        except Exception as e:
            logger.error(f"Error initializing Google AI client: {e}")
            logger.error("Model operations will be unavailable.")
            self.client = None
            return False

    def is_client_ready(self) -> bool:
        """Check if the Google AI client is initialized and ready."""
        return self.client is not None

    def validate_api_key(self, api_key: str) -> bool:
        """Validate a Google AI API key by attempting to list models."""
        if not HAS_GENAI:
            logger.error("Google SDK not found, cannot validate API key.")
            return False
        try:
            temp_client = genai.Client(api_key=api_key)
            list(temp_client.models.list()) # Force iteration using new SDK structure
            logger.info("API key validation successful.")
            return True
        except (google_exceptions.PermissionDenied, google_exceptions.Unauthenticated):
            logger.error("API key validation failed: Invalid key or insufficient permissions.")
            return False
        except Exception as e:
            logger.error(f"API key validation encountered an unexpected error: {e}")
            return False

    def get_tunable_models(self) -> List[str]:
        """Returns a list of known tunable model names."""
        # In a more advanced version, this could try to fetch dynamically
        # from self.client.list_models() and filter, but the API doesn't
        # currently expose a simple "is_tunable" flag reliably.
        return self.TUNABLE_MODELS


    def train_model(self,
                   training_data: List[Dict[str, str]],
                   base_model: Optional[str] = None,
                   display_name: Optional[str] = None,
                   epochs: Optional[int] = None,
                   batch_size: Optional[int] = None,
                   learning_rate_multiplier: Optional[float] = None) -> Optional[str]:
        """
        Starts tuning a Gemini model using the provided training data.

        Args:
            training_data: List of dictionaries with 'text_input' and 'output'.
            base_model: Base model identifier (e.g., 'models/gemini-1.0-pro-001').
            display_name: Custom name for the tuned model in the UI.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            learning_rate_multiplier: Multiplier for the base learning rate.

        Returns:
            The full name (ID) of the tuned model if training completes successfully (e.g., 'tunedModels/abc-123'),
            otherwise None. Returns immediately after starting the job. Monitoring happens separately or later.
        """
        if not self.is_client_ready():
            logger.error("Cannot train model: Google AI client not initialized.")
            return None
        if not training_data:
            logger.error("Cannot train model: No training data provided.")
            return None

        base_model_name = base_model or self.config.get('base_model')
        job_display_name = display_name or f"Tuned_{Path(base_model_name).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        train_epochs = epochs or self.config.get('epochs')
        train_batch_size = batch_size or self.config.get('batch_size')
        train_lr_multiplier = learning_rate_multiplier or self.config.get('learning_rate_multiplier')

        # Validate base model is tunable (basic check)
        if base_model_name not in self.TUNABLE_MODELS:
             logger.warning(f"Base model '{base_model_name}' is not in the known list of tunable models ({self.TUNABLE_MODELS}). Training might fail.")
             # Optionally add a confirmation step here in interactive/GUI modes

        try:
            logger.info(f"Starting tuning job '{job_display_name}' with {len(training_data)} examples...")
            logger.info(f"  Base Model: {base_model_name}")
            logger.info(f"  Hyperparameters: epochs={train_epochs}, batch_size={train_batch_size}, lr_multiplier={train_lr_multiplier}")

            # Define the tuning task using SDK types
            # Convert training data to the format expected by the new SDK
            try:
                # First, log the structure of the training data for debugging
                if training_data and len(training_data) > 0:
                    logger.info(f"Sample training data keys: {list(training_data[0].keys())}")
                    logger.info(f"Sample training data first item: {training_data[0]}")
                
                # Try to determine the correct keys
                input_key = None
                output_key = None
                
                # Check the first item to determine keys
                if training_data and len(training_data) > 0:
                    sample = training_data[0]
                    # Check common input key names
                    for key in ["input", "text_input", "text", "prompt"]:
                        if key in sample:
                            input_key = key
                            break
                    # Check common output key names
                    for key in ["output", "response", "completion", "answer"]:
                        if key in sample:
                            output_key = key
                            break
                
                # If we couldn't determine keys, use defaults
                input_key = input_key or "input"
                output_key = output_key or "output"
                
                logger.info(f"Using input_key='{input_key}' and output_key='{output_key}' for training data")
                
                tuning_dataset = types.TuningDataset(
                    examples=[
                        types.TuningExample(
                            text_input=example.get(input_key, ""),
                            output=example.get(output_key, "")
                        )
                        for example in training_data
                    ]
                )
            except Exception as e:
                logger.error(f"Error preparing training data: {e}")
                return None
            
            # Start the tuning job using the new SDK structure
            try:
                tuning_job = self.client.tunings.tune(
                    base_model=base_model_name,
                    training_dataset=tuning_dataset,
                    config=types.CreateTuningJobConfig(
                        epoch_count=train_epochs,
                        batch_size=train_batch_size,
                        learning_rate=train_lr_multiplier,
                        tuned_model_display_name=job_display_name
                    )
                )
                
                if tuning_job is None:
                    logger.error("Tuning job creation failed and returned None")
                    return None
                    
                # Check if tuned_model exists
                if not hasattr(tuning_job, 'tuned_model') or tuning_job.tuned_model is None:
                    logger.error("Tuning job created but tuned_model is None")
                    return job_display_name  # Return display name as fallback
                
                logger.info(f"Tuning job submitted. Operation Name: {tuning_job.tuned_model.model}")
                logger.info("Training is running in the background on Google Cloud.")
                logger.info("Use 'list models' or the Model Manager UI to check status.")
                logger.info("You can also monitor via Google Cloud Console if needed.")
            except Exception as e:
                logger.error(f"Error during tuning job creation: {e}")
                return None

            # --- Polling Logic (Optional - Can be done separately) ---
            # The SDK's `tuning_operation.result()` blocks until completion.
            # For a non-blocking approach or progress, you need polling.
            # We'll return the operation name and let other parts handle monitoring/results.

            # If you NEED to block and wait here:
            # print("Waiting for tuning job to complete... (This can take hours)")
            # try:
            #     # result() blocks until done, raises if fails, times out after default period
            #     tuned_model_result = tuning_operation.result()
            #     logger.info(f"Tuning job '{job_display_name}' completed successfully!")
            #     logger.info(f"Tuned Model Name (ID): {tuned_model_result.name}")
            #     return tuned_model_result.name
            # except Exception as e:
            #     logger.error(f"Tuning job '{job_display_name}' failed or timed out.")
            #     # Attempt to get final status for error details
            #     try:
            #         final_status = tuning_operation.metadata
            #         logger.error(f"Job State: {final_status.state if hasattr(final_status, 'state') else 'Unknown'}")
            #         # Look for error details if the SDK provides them in metadata
            #     except Exception as status_e:
            #          logger.error(f"Could not retrieve final status details: {status_e}")
            #     return None

            # For non-blocking: return the tuned model name from the job
            # This will allow the caller to use this ID to check status later
            try:
                return tuning_job.tuned_model.model
            except Exception as e:
                logger.error(f"Error accessing tuned model information: {e}")
                return job_display_name  # Return display name as fallback


        except google_exceptions.InvalidArgument as e:
             logger.error(f"Failed to start tuning job: Invalid Argument. Check parameters and data. Detail: {e}")
             return None
        except google_exceptions.PermissionDenied as e:
             logger.error(f"Failed to start tuning job: Permission Denied. Check API key/project permissions. Detail: {e}")
             return None
        except Exception as e:
            logger.error(f"An unexpected error occurred starting the tuning job: {e}")
            return None

    def get_tuning_job_status(self, tuned_model_name: str) -> Optional[Dict[str, Any]]:
         """Gets the status of a tuning job by its tuned model name."""
         if not self.is_client_ready():
            logger.error("Cannot get status: Google AI client not initialized.")
            return None
         try:
            # The API uses tunings.get_tuned_model to check status in the new SDK
            model_info = self.client.tunings.get(name=tuned_model_name)
            return {
                "name": model_info.name,
                "display_name": model_info.display_name,
                "state": str(model_info.state), # Convert enum to string
                "create_time": model_info.create_time,
                "update_time": model_info.update_time,
                # Add other relevant fields if needed
            }
         except google_exceptions.NotFound:
             logger.warning(f"Tuned model '{tuned_model_name}' not found.")
             return None
         except Exception as e:
             logger.error(f"Error getting status for tuned model '{tuned_model_name}': {e}")
             return None


    def list_models(self, only_tuned: bool = True) -> List[Dict[str, Any]]:
        """Lists models available via the API."""
        if not self.is_client_ready():
            logger.error("Cannot list models: Google AI client not initialized.")
            return []

        try:
            logger.debug(f"Listing {'tuned models only' if only_tuned else 'all models'}...")
            # Update to use the new SDK structure
            if only_tuned:
                # For tuned models in the new SDK
                models_iterator = self.client.tunings.list()
            else:
                # For all models in the new SDK
                models_iterator = self.client.models.list()
            
            model_info_list = []

            # Use tqdm for progress if listing all models (can be many)
            model_iter_display = tqdm(models_iterator, desc="Fetching models") if HAS_TQDM and not only_tuned else models_iterator

            for model in model_iter_display:
                info = {
                    "name": model.name, # Full ID (e.g., models/gemini..., tunedModels/...)
                    "display_name": getattr(model, 'display_name', Path(model.name).name), # Use display_name if available
                    "description": getattr(model, 'description', 'N/A'),
                    "version": getattr(model, 'version', 'N/A'),
                    # Fields specific to tuned models
                    "state": str(getattr(model, 'state', 'N/A')), # Convert enum if present
                    "base_model": getattr(model, 'base_model', getattr(model, 'source_model', 'N/A')), # Use base_model or source_model
                    "create_time": getattr(model, 'create_time', 'N/A'),
                    "update_time": getattr(model, 'update_time', 'N/A'),
                    # General model fields
                    "input_token_limit": getattr(model, 'input_token_limit', 'N/A'),
                    "output_token_limit": getattr(model, 'output_token_limit', 'N/A'),
                    "supported_generation_methods": getattr(model, 'supported_generation_methods', []),
                }
                # Add a flag indicating if it's a tuned model
                info["is_tuned"] = model.name.startswith("tunedModels/")

                # Filter based on 'only_tuned' if not already done by the API call
                # (client.list_tuned_models already filters, this is belt-and-suspenders)
                if only_tuned and not info["is_tuned"]:
                    continue

                model_info_list.append(info)

            if HAS_TQDM and not only_tuned: model_iter_display.close()

            logger.info(f"Found {len(model_info_list)} {'tuned' if only_tuned else ''} models.")
            # Sort tuned models by creation time, newest first
            if only_tuned:
                 model_info_list.sort(key=lambda x: x.get('create_time'), reverse=True)

            return model_info_list
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def delete_model(self, model_name: str) -> bool:
        """Delete a tuned model."""
        if not self.is_client_ready():
            logger.error("Cannot delete model: Google AI client not initialized.")
            return False
    
        if not model_name or not model_name.startswith("tunedModels/"):
            logger.error(f"Invalid tuned model name for deletion: '{model_name}'. Must start with 'tunedModels/'.")
            return False
    
        try:
            logger.warning(f"Attempting to delete tuned model: {model_name}")
            
            # Use direct REST API approach with requests library
            import requests
            
            # Get the API key from environment or client
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key and hasattr(self.client, "_credentials"):
                # Try to extract from client credentials
                try:
                    api_key = self.client._credentials.token
                except:
                    pass
                    
            if not api_key:
                logger.error("Cannot delete model: API key not available")
                return False
                
            # Make a DELETE request to the Gemini API
            url = f"https://generativelanguage.googleapis.com/v1/{model_name}"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.delete(url, headers=headers)
            
            if response.status_code in (200, 204, 404):
                logger.info(f"Successfully deleted model: {model_name}")
                return True
            else:
                logger.error(f"Failed to delete model: {response.status_code} - {response.text}")
                return False
                
        except google_exceptions.NotFound:
            logger.warning(f"Tuned model '{model_name}' not found. Assuming already deleted.")
            return True
        except google_exceptions.PermissionDenied as e:
            logger.error(f"Permission denied when deleting model '{model_name}': {e}")
            return False
            
        except Exception as e:
            logger.error(f"Error deleting tuned model '{model_name}': {e}")
            return False





    def test_model(self, model_name: str, test_input: str, generation_config_override: Optional[Dict] = None) -> Optional[str]:
        """
        Tests a model (base or tuned) with sample input using generate_content.

        Args:
            model_name: Full name/ID of the model (e.g., 'models/gemini...', 'tunedModels/abc-123').
            test_input: The input text prompt.
            generation_config_override: Optional dictionary for generation parameters
                                         (e.g., {"temperature": 0.5, "max_output_tokens": 100}).

        Returns:
            The generated text response, or None if an error occurs.
        """
        if not self.is_client_ready():
            logger.error("Cannot test model: Google AI client not initialized.")
            return None
        if not model_name:
             logger.error("Cannot test model: No model name provided.")
             return None
        if not test_input:
             logger.error("Cannot test model: No test input provided.")
             return None

        try:
            # Define default generation config (can be customized)
            default_config = {
                "temperature": 0.7, # Default is often higher, adjust as needed
                "max_output_tokens": 1024,
                "top_p": 0.95,
                "top_k": 40
            }
            # Merge overrides
            final_config_dict = default_config.copy()
            if generation_config_override:
                final_config_dict.update(generation_config_override)

            # Convert dict to the SDK's GenerationConfig object
            gen_config = types.GenerationConfig(**final_config_dict)


            logger.info(f"Sending test input to model: {model_name}")
            logger.debug(f"Input: '{test_input[:100]}...'")
            logger.debug(f"Generation Config: {final_config_dict}")

            # Use the client's models.generate_content method directly
            response = self.client.models.generate_content(
                model=model_name,
                contents=test_input,
                generation_config=gen_config
                # safety_settings= # Optional: Add safety settings if needed
            )

            # Process response - check for empty or blocked content
            if not response.candidates:
                 logger.warning(f"Model '{model_name}' returned no candidates. Response might be empty or blocked.")
                 # Check finish reason if available
                 try:
                      finish_reason = response.prompt_feedback.block_reason
                      if finish_reason:
                           logger.warning(f"Content blocked. Reason: {finish_reason}")
                      else:
                           # Check safety ratings
                           for rating in response.prompt_feedback.safety_ratings:
                                if rating.probability != types.SafetyRating.Probability.NEGLIGIBLE:
                                     logger.warning(f"Potential safety issue detected: Category={rating.category}, Probability={rating.probability}")
                 except Exception:
                      logger.warning("Could not determine reason for empty response.")
                 return None # Or return a specific message like "[Blocked Content]"

            # Assuming the first candidate is the primary one
            generated_text = response.text
            logger.info(f"Successfully received response from {model_name}.")
            logger.debug(f"Output: '{generated_text[:100]}...'")
            return generated_text

        except google_exceptions.NotFound:
             logger.error(f"Model '{model_name}' not found.")
             return None
        except google_exceptions.InvalidArgument as e:
            logger.error(f"Invalid argument testing model '{model_name}'. Check input or config. Detail: {e}")
            return None
        except google_exceptions.PermissionDenied as e:
             logger.error(f"Permission denied while testing model '{model_name}'. Detail: {e}")
             return None
        except Exception as e:
            # Catch potential errors from response processing (e.g., response.text)
            logger.error(f"Error testing model '{model_name}': {e}")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 logger.error(f"Content likely blocked. Reason: {response.prompt_feedback.block_reason}")
            return None


    # --- Chat Passthrough Methods ---

    def create_chat(self, model_name: Optional[str] = None) -> Optional[str]:
        """Creates a new chat session using the ChatManager."""
        if not self.chat_manager: return None
        return self.chat_manager.create_chat(model_name)

    def send_message(self, chat_id: str, message: str, stream: bool = False) -> Union[str, Generator[str, None, None], None]:
        """Sends a message to a chat session using the ChatManager."""
        if not self.chat_manager: return None
        return self.chat_manager.send_message(chat_id, message, stream)

    def get_chat_history(self, chat_id: str) -> List[Dict[str, str]]:
        """Gets chat history using the ChatManager."""
        if not self.chat_manager: return []
        return self.chat_manager.get_history(chat_id)

    def list_chats(self) -> List[Dict[str, Any]]:
        """Lists active chats using the ChatManager."""
        if not self.chat_manager: return []
        return self.chat_manager.list_chats()

    def delete_chat(self, chat_id: str) -> bool:
        """Deletes a chat session using the ChatManager."""
        if not self.chat_manager: return False
        return self.chat_manager.delete_chat(chat_id)


# --- Interactive CLI Class ---

class InteractiveCLI:
    """Interactive command-line interface for Gemini Tuner."""

    def __init__(self, config: Config, processor: DataProcessor, model_manager: ModelManager):
        self.config = config
        self.processor = processor
        self.model_manager = model_manager
        self.training_data: List[Dict[str, str]] = [] # Store loaded/processed data

    def run(self) -> None:
        """Run the interactive CLI workflow."""
        self._print_header()

        # Step 1: Configuration & API Key Check
        if not self._setup_configuration():
            logger.info("Exiting due to configuration issues.")
            return

        while True:
             action = self._main_menu()
             if action == 'process':
                  self._run_processing()
             elif action == 'load':
                  self._run_load_data()
             elif action == 'train':
                  self._run_training()
             elif action == 'manage':
                  self._run_model_management()
             elif action == 'test':
                  self._run_testing()
             elif action == 'chat':
                  self._run_chat()
             elif action == 'config':
                 self._setup_configuration(force_update=True) # Allow re-config
             elif action == 'quit':
                  break

        print("\n👋 Goodbye!")

    def _print_header(self) -> None:
        print("\n" + "="*50)
        print("🔮 Welcome to Gemini Tuner - Interactive Mode 🔮")
        print("="*50 + "\n")

    def _main_menu(self) -> str:
         """Display the main menu and get user choice."""
         print("\n--- Main Menu ---")
         print("Current Data: ", f"{len(self.training_data)} examples loaded." if self.training_data else "No data loaded.")
         print("API Status:   ", "Ready" if self.model_manager.is_client_ready() else "Unavailable (No valid API Key)")
         print("\nChoose an action:")
         print("  [P] Process raw text file(s) into training data")
         print("  [L] Load existing training data (.jsonl)")
         print("  [T] Train a model (requires loaded data & API key)")
         print("  [M] Manage tuned models (List, Delete) (requires API key)")
         print("  [S] Test a model (base or tuned) (requires API key)")
         print("  [H] Chat with a model (requires API key)")
         print("  [C] Configure settings / API Key")
         print("  [Q] Quit")

         while True:
             choice = input("> Enter choice: ").strip().upper()
             if choice == 'P': return 'process'
             if choice == 'L': return 'load'
             if choice == 'T': return 'train'
             if choice == 'M': return 'manage'
             if choice == 'S': return 'test'
             if choice == 'H': return 'chat'
             if choice == 'C': return 'config'
             if choice == 'Q': return 'quit'
             print("Invalid choice, please try again.")


    def _setup_configuration(self, force_update=False) -> bool:
        """Set up configuration and validate API key interactively."""
        print("\n--- Configuration Setup ---")

        # Initial check or forced update
        if force_update or not self.model_manager.is_client_ready():
            current_key = self.config.get('api_key')
            print(f"Current API Key Status: {'Valid' if self.model_manager.is_client_ready() else 'Not Set or Invalid'}")
            if self.config.config_file:
                print(f"Config loaded from: {self.config.config_file}")

            update_key = force_update or self._confirm("Update or verify API Key?", default=(not current_key))
            if update_key:
                new_key = self._prompt_password("Enter your Google AI API key (leave blank to keep current)")
                if new_key:
                    print("Validating new API key...")
                    if self.model_manager.validate_api_key(new_key):
                        self.config.update(api_key=new_key)
                        # Re-initialize the client with the new key
                        self.model_manager._initialize_genai()
                        if self._confirm("Save this valid API key to your configuration file?", default=True):
                             self.config.save_config()
                    else:
                        print("New API key is invalid. Keeping previous configuration.")
                elif not current_key:
                    print("No API key provided. Model operations remain disabled.")

        # Always check final status
        if not self.model_manager.is_client_ready():
            print("\nWARNING: Google AI API key is not configured or invalid.")
            print("         Model tuning, testing, listing, and chat are disabled.")
            if not force_update: # Only ask to quit if it wasn't a forced config update
                if not self._confirm("Continue with file processing only?", default=True):
                    return False # Signal to exit
        else:
             print("\nGoogle AI API Key is configured and valid.")

        # Optionally configure other settings
        if self._confirm("Review/edit other default settings (model, format)?", default=False):
             self._edit_other_configs()

        return True # Configuration setup completed (or user chose to continue without API key)

    def _edit_other_configs(self):
        """Allow interactive editing of non-API key configs."""
        print("\n--- Edit Default Settings ---")
        # Base Model
        current_base = self.config.get('base_model')
        print(f"\nAvailable Tunable Base Models: {self.model_manager.TUNABLE_MODELS}")
        new_base = self._prompt("Default Base Model for tuning", default=current_base)
        if new_base != current_base and new_base in self.model_manager.TUNABLE_MODELS:
            self.config.update(base_model=new_base)
        elif new_base != current_base:
             print(f"Warning: '{new_base}' not in known tunable list. Keeping '{current_base}'.")

        # Training Params
        new_epochs = self._prompt("Default Epochs", default=str(self.config.get('epochs')))
        new_batch = self._prompt("Default Batch Size", default=str(self.config.get('batch_size')))
        new_lr = self._prompt("Default Learning Rate Multiplier", default=str(self.config.get('learning_rate_multiplier')))
        self.config.update(epochs=new_epochs, batch_size=new_batch, learning_rate_multiplier=new_lr) # update handles conversion/validation

        # Format Params
        new_delim = self._prompt("Default Example Delimiter", default=self.config.get('delimiter'))
        new_in_prefix = self._prompt("Default Input Prefix", default=self.config.get('input_prefix'))
        new_out_prefix = self._prompt("Default Output Prefix", default=self.config.get('output_prefix'))
        self.config.update(delimiter=new_delim, input_prefix=new_in_prefix, output_prefix=new_out_prefix)

        if self._confirm("Save these updated defaults to your configuration file?", default=True):
             self.config.save_config()


    def _run_processing(self):
        """Handle the file processing workflow."""
        print("\n--- Process Raw Text Files ---")
        file_paths = self._select_input_files()
        if not file_paths:
            print("No input files selected.")
            return

        delimiter = self._prompt("Example delimiter", default=self.config.get('delimiter'))
        input_prefix = self._prompt("Input prefix", default=self.config.get('input_prefix'))
        output_prefix = self._prompt("Output prefix", default=self.config.get('output_prefix'))

        try:
            print("Processing...")
            processed_data, valid_count, invalid_count = self.processor.process_multiple_files(
                file_paths,
                delimiter=delimiter,
                input_prefix=input_prefix,
                output_prefix=output_prefix
            )

            if not processed_data:
                 print("\nNo valid examples were extracted from the files.")
                 return

            print(f"\nProcessing complete: {valid_count} valid examples, {invalid_count} skipped.")

            if self._confirm("Preview processed data?", default=True):
                self.processor.preview_training_data(processed_data)

            if self._confirm("Save processed data to JSONL file?", default=True):
                default_out = f"training_data_{datetime.now().strftime('%Y%m%d')}.jsonl"
                output_file = self._prompt("Output file path (.jsonl)", default=default_out)
                if output_file:
                    try:
                        self.processor.save_training_data(processed_data, output_file)
                        # Ask if user wants to load this data now
                        if self._confirm("Load this saved data for training/testing?", default=True):
                             self.training_data = processed_data
                             print(f"{len(self.training_data)} examples loaded into memory.")
                    except Exception as e:
                         print(f"Error saving data: {e}")
            elif self._confirm("Load processed data directly into memory (without saving)?", default=False):
                 self.training_data = processed_data
                 print(f"{len(self.training_data)} examples loaded into memory.")


        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during processing: {e}")

    def _run_load_data(self):
         """Handle loading data from JSONL."""
         print("\n--- Load Training Data (.jsonl) ---")
         input_file = self._prompt("Enter path to training data file (.jsonl)")
         if not input_file:
              return
         try:
              self.training_data = self.processor.load_training_data(input_file)
              print(f"Successfully loaded {len(self.training_data)} examples.")
              if self._confirm("Preview loaded data?", default=False):
                   self.processor.preview_training_data(self.training_data)
         except FileNotFoundError:
              print(f"Error: File not found at '{input_file}'")
         except Exception as e:
              print(f"Error loading data: {e}")


    def _run_training(self):
        """Handle the model training workflow."""
        print("\n--- Train a Model ---")
        if not self.model_manager.is_client_ready():
            print("API Key not configured. Cannot train models.")
            return
        if not self.training_data:
            print("No training data loaded. Please process or load data first.")
            return

        print(f"Using {len(self.training_data)} loaded examples.")
        if len(self.training_data) < 10:
             if not self._confirm("Dataset size is very small (<10). Training might fail. Continue anyway?", default=False):
                  return

        # Get parameters
        print(f"\nAvailable Tunable Base Models: {self.model_manager.TUNABLE_MODELS}")
        base_model = self._prompt("Base model to tune", default=self.config.get('base_model'))
        if base_model not in self.model_manager.TUNABLE_MODELS:
             if not self._confirm(f"Warning: '{base_model}' not in known tunable list. Continue?", default=False):
                  return

        default_display_name = f"Tuned_{Path(base_model).name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        display_name = self._prompt("Display name for tuned model", default=default_display_name)

        epochs = int(self._prompt("Number of epochs", default=str(self.config.get('epochs'))))
        batch_size = int(self._prompt("Batch size", default=str(self.config.get('batch_size'))))
        lr_multiplier = float(self._prompt("Learning rate multiplier", default=str(self.config.get('learning_rate_multiplier'))))

        if self._confirm("Start training with these settings?", default=True):
            print("\nSubmitting training job...")
            # Train model (now returns quickly after submission)
            submitted_job_ref = self.model_manager.train_model(
                training_data=self.training_data,
                base_model=base_model,
                display_name=display_name,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate_multiplier=lr_multiplier
            )

            if submitted_job_ref:
                print("\nTraining job submitted successfully!")
                print(f"Model Display Name: {display_name}")
                print("The job is running in the background on Google Cloud.")
                print("Use the 'Manage tuned models' option [M] to check its status later.")
                # Potential TODO: Offer to poll status here, but might block CLI
            else:
                print("\nFailed to submit training job. Check logs for errors.")


    def _run_model_management(self):
         """Handle listing and deleting tuned models."""
         print("\n--- Manage Tuned Models ---")
         if not self.model_manager.is_client_ready():
             print("API Key not configured. Cannot manage models.")
             return

         while True:
             print("\nModel Management Actions:")
             print("  [L] List tuned models")
             print("  [D] Delete a tuned model")
             print("  [S] Check status of a specific model")
             print("  [B] Back to main menu")
             choice = input("> Enter choice: ").strip().upper()

             if choice == 'L':
                  self._list_tuned_models()
             elif choice == 'D':
                  self._delete_tuned_model()
             elif choice == 'S':
                  self._check_model_status()
             elif choice == 'B':
                  break
             else:
                  print("Invalid choice.")

    def _list_tuned_models(self):
        print("\nFetching tuned models...")
        models = self.model_manager.list_models(only_tuned=True)
        if not models:
            print("No tuned models found for this project.")
            return

        print("\n--- Your Tuned Models ---")
        for i, model in enumerate(models):
            create_time_str = model.get('create_time', 'N/A')
            if hasattr(create_time_str, 'strftime'): # Check if it's a datetime object
                create_time_str = create_time_str.strftime('%Y-%m-%d %H:%M')
            print(f"\n{i+1}. Display Name: {model.get('display_name', 'N/A')}")
            print(f"   Model Name (ID): {model.get('name', 'N/A')}")
            print(f"   Base Model:      {model.get('base_model', 'N/A')}")
            print(f"   State:           {model.get('state', 'N/A').upper()}")
            print(f"   Created:         {create_time_str}")
        print("--- End of List ---")

    def _delete_tuned_model(self):
         models = self.model_manager.list_models(only_tuned=True)
         if not models:
             print("No tuned models found to delete.")
             return

         self._list_tuned_models() # Show list first
         while True:
             try:
                 choice_str = input("\nEnter the number of the model to delete (or 0 to cancel): ")
                 choice_idx = int(choice_str) - 1
                 if choice_idx == -1: # User entered 0
                      print("Deletion cancelled.")
                      return
                 if 0 <= choice_idx < len(models):
                      model_to_delete = models[choice_idx]
                      break
                 else:
                      print("Invalid number.")
             except ValueError:
                  print("Invalid input. Please enter a number.")

         model_name = model_to_delete.get('name')
         display_name = model_to_delete.get('display_name')
         print(f"\nYou selected: {display_name} ({model_name})")
         if self._confirm(f"ARE YOU SURE you want to permanently delete this model?", default=False):
              print(f"Deleting {model_name}...")
              success = self.model_manager.delete_model(model_name)
              if success:
                   print("Model deleted successfully.")
              else:
                   print("Failed to delete model. Check logs.")
         else:
              print("Deletion cancelled.")

    def _check_model_status(self):
        model_name = self._prompt("Enter the full Model Name (ID) to check (e.g., tunedModels/abc-123)")
        if not model_name or not model_name.startswith("tunedModels/"):
            print("Invalid input. Please provide the full tuned model name.")
            return

        print(f"Checking status for {model_name}...")
        status = self.model_manager.get_tuning_job_status(model_name)

        if status:
             print("\n--- Model Status ---")
             print(f"  Display Name: {status.get('display_name', 'N/A')}")
             print(f"  Model Name (ID): {status.get('name', 'N/A')}")
             state = status.get('state', 'UNKNOWN').upper()
             print(f"  State:           {state}")
             create_time = status.get('create_time', 'N/A')
             update_time = status.get('update_time', 'N/A')
             if hasattr(create_time, 'strftime'): create_time = create_time.strftime('%Y-%m-%d %H:%M:%S')
             if hasattr(update_time, 'strftime'): update_time = update_time.strftime('%Y-%m-%d %H:%M:%S')
             print(f"  Created:         {create_time}")
             print(f"  Last Updated:    {update_time}")

             if "FAILED" in state:
                  print("\n  NOTE: Job failed. Check Google Cloud logs for details.")
             elif "ACTIVE" in state:
                   print("\n  NOTE: Model is ready and active for use.")
             elif "CREATING" in state:
                   print("\n  NOTE: Training is likely still in progress.")
             print("--- End Status ---")

        else:
             # Error message already logged by get_tuning_job_status
             pass


    def _run_testing(self):
        """Handle testing a model."""
        print("\n--- Test a Model ---")
        if not self.model_manager.is_client_ready():
            print("API Key not configured. Cannot test models.")
            return

        # Choose model: List base and tuned
        print("\nAvailable Models for Testing:")
        print("  Base Models:")
        base_models = self.model_manager.get_tunable_models() # Or list all generative models?
        for bm in base_models:
             print(f"    - {bm}")

        tuned_models = self.model_manager.list_models(only_tuned=True)
        active_tuned = [m for m in tuned_models if m.get('state', '').upper() == 'ACTIVE']
        if active_tuned:
            print("\n  Active Tuned Models:")
            for i, tm in enumerate(active_tuned):
                print(f"    [{i+1}] {tm.get('display_name')} ({tm.get('name')})")

        model_to_test = None
        while not model_to_test:
             choice = input("\nEnter base model name, tuned model number, or full ID: ").strip()
             if not choice: continue
             # Check if it's a number for tuned list
             try:
                 idx = int(choice) - 1
                 if 0 <= idx < len(active_tuned):
                      model_to_test = active_tuned[idx].get('name')
                      print(f"Selected tuned model: {active_tuned[idx].get('display_name')}")
                      break
             except ValueError:
                  pass # Not a number, treat as name/ID

             # Check if it's a known base model or full ID
             if choice in base_models or choice.startswith("models/") or choice.startswith("tunedModels/"):
                  model_to_test = choice
                  print(f"Selected model: {model_to_test}")
                  break
             else:
                  print("Invalid selection. Please enter a valid base model name, tuned model number, or full ID.")


        # Get input
        print("\nEnter text input for the model (Ctrl+D or Ctrl+Z on Windows then Enter to finish):")
        lines = []
        while True:
            try:
                line = input()
                lines.append(line)
            except EOFError:
                break
        test_input = "\n".join(lines)

        if not test_input:
            print("No input provided.")
            return

        # Get generation config (optional)
        gen_config = {}
        if self._confirm("Customize generation parameters (temperature, etc.)?", default=False):
             temp = self._prompt("Temperature (e.g., 0.7)", default="0.7")
             max_tokens = self._prompt("Max Output Tokens (e.g., 1024)", default="1024")
             try:
                  gen_config['temperature'] = float(temp)
                  gen_config['max_output_tokens'] = int(max_tokens)
                  # Could add top_p, top_k here too
             except ValueError:
                  print("Invalid number format for generation parameters. Using defaults.")
                  gen_config = None


        print("\nGenerating response...")
        response = self.model_manager.test_model(model_to_test, test_input, gen_config)

        if response is not None: # Check for None explicitly, as "" is a valid (empty) response
            print("\n--- Model Response ---")
            print(response)
            print("--- End Response ---")

            # Save response?
            if self._confirm("Save response to a file?", default=False):
                 out_file = self._prompt("Enter output file name", default="model_response.txt")
                 if out_file:
                      try:
                           with open(out_file, "w", encoding="utf-8") as f:
                                f.write(response)
                           print(f"Response saved to {out_file}")
                      except IOError as e:
                           print(f"Error saving file: {e}")
        else:
            print("\nFailed to get response from the model. Check logs for details (e.g., content blocked).")


    def _run_chat(self):
         """Handle interactive chat with a model."""
         print("\n--- Chat with Gemini ---")
         if not self.model_manager.is_client_ready():
             print("API Key not configured. Cannot start chat.")
             return

         # Choose model (similar logic to testing)
         print("\nChoose a model for chat:")
         print("  Base Models:")
         base_models = self.model_manager.get_tunable_models()
         for bm in base_models:
              print(f"    - {bm}")
         tuned_models = self.model_manager.list_models(only_tuned=True)
         active_tuned = [m for m in tuned_models if m.get('state', '').upper() == 'ACTIVE']
         if active_tuned:
             print("\n  Active Tuned Models:")
             for i, tm in enumerate(active_tuned):
                 print(f"    [{i+1}] {tm.get('display_name')} ({tm.get('name')})")

         model_for_chat = None
         while not model_for_chat:
              choice = input("\nEnter base model name, tuned model number, or full ID: ").strip()
              if not choice: continue
              try:
                  idx = int(choice) - 1
                  if 0 <= idx < len(active_tuned):
                       model_for_chat = active_tuned[idx].get('name')
                       print(f"Starting chat with tuned model: {active_tuned[idx].get('display_name')}")
                       break
              except ValueError: pass
              if choice in base_models or choice.startswith("models/") or choice.startswith("tunedModels/"):
                   model_for_chat = choice
                   print(f"Starting chat with model: {model_for_chat}")
                   break
              else:
                   print("Invalid selection.")

         # Create chat session
         chat_id = self.model_manager.create_chat(model_name=model_for_chat)
         if not chat_id:
              print("Failed to start chat session.")
              return

         print("\nChat started. Type your message and press Enter.")
         print("Type '/quit' or '/exit' to end the chat.")
         print("Type '/history' to see the conversation.")
         print("-" * 20)

         while True:
             try:
                 user_input = input("You: ")
                 if not user_input:
                     continue
                 if user_input.lower() in ['/quit', '/exit']:
                     break
                 if user_input.lower() == '/history':
                      history = self.model_manager.get_chat_history(chat_id)
                      print("\n--- Chat History ---")
                      for msg in history:
                           role = "Gemini" if msg['role'] == 'model' else "You"
                           print(f"{role}: {msg['content']}")
                      print("--------------------\n")
                      continue

                 # Send message and stream response
                 print("Gemini: ", end="", flush=True)
                 response_stream = self.model_manager.send_message(chat_id, user_input, stream=True)

                 if response_stream:
                     full_response = ""
                     try:
                         for chunk in response_stream:
                             print(chunk, end="", flush=True)
                             full_response += chunk
                         print() # Newline after Gemini finishes
                     except Exception as e:
                          print(f"\n[Error during response streaming: {e}]")
                 else:
                      print("\n[No response or error occurred]")

             except EOFError: # Handle Ctrl+D
                  print("\nExiting chat.")
                  break
             except KeyboardInterrupt: # Handle Ctrl+C
                  print("\nExiting chat.")
                  break

         # Clean up chat session
         self.model_manager.delete_chat(chat_id)
         print("Chat session closed.")


    def _select_input_files(self) -> List[str]:
        """Select input files interactively."""
        file_paths = []
        print("Enter paths to input text file(s). Press Enter after each file.")
        print("Enter an empty line when finished.")
        while True:
            try:
                 file_path = input(f"  File {len(file_paths) + 1}: ").strip()
                 if not file_path:
                     if file_paths: break
                     else: print("Please enter at least one file path.")
                 elif Path(file_path).is_file():
                     file_paths.append(file_path)
                     print(f"    Added: {file_path}")
                 else:
                     print(f"    Error: File not found at '{file_path}'")
            except EOFError: # Handle Ctrl+D
                 break
        return file_paths

    def _prompt(self, message: str, default: Optional[str] = None) -> str:
        """Prompt for user input with optional default value."""
        prompt_text = f"> {message}"
        if default is not None:
            prompt_text += f" [{default}]"
        prompt_text += ": "
        user_input = input(prompt_text).strip()
        return user_input if user_input else (default if default is not None else "")

    def _prompt_password(self, message: str) -> str:
        """Prompt for password input (without echo)."""
        prompt_text = f"> {message}: "
        try:
            return getpass.getpass(prompt_text).strip()
        except (ImportError, EOFError):
            # Fallback if getpass is not available or running in odd env
            print("\nWarning: Could not hide password input.")
            return input(prompt_text).strip()


    def _confirm(self, message: str, default: bool = False) -> bool:
        """Prompt for confirmation (Yes/No)."""
        default_str = "[Y/n]" if default else "[y/N]"
        while True:
            response = input(f"> {message} {default_str}: ").strip().lower()
            if not response:
                return default
            if response.startswith('y'):
                return True
            if response.startswith('n'):
                return False
            print("Please answer 'yes' or 'no'.")


# --- GUI Class ---

class GUI:
    """Graphical user interface for Gemini Tuner."""

    # Define keys for UI elements
    KEY_NAV_SETUP = 'NAV_SETUP'
    KEY_NAV_PROCESS = 'NAV_PROCESS'
    KEY_NAV_TRAIN = 'NAV_TRAIN'
    KEY_NAV_TEST = 'NAV_TEST'
    KEY_NAV_MANAGE = 'NAV_MANAGE'
    KEY_NAV_CHAT = 'NAV_CHAT'

    KEY_CONTENT_SETUP = 'CONTENT_SETUP'
    KEY_CONTENT_PROCESS = 'CONTENT_PROCESS'
    KEY_CONTENT_TRAIN = 'CONTENT_TRAIN'
    KEY_CONTENT_TEST = 'CONTENT_TEST'
    KEY_CONTENT_MANAGE = 'CONTENT_MANAGE'
    KEY_CONTENT_CHAT = 'CONTENT_CHAT'

    KEY_STATUS = 'STATUS_BAR'

    # Setup Tab Keys
    KEY_SETUP_API_KEY = 'SETUP_API_KEY'
    KEY_SETUP_TEST_API = 'SETUP_TEST_API'
    KEY_SETUP_API_STATUS = 'SETUP_API_STATUS'
    KEY_SETUP_SAVE_CONFIG = 'SETUP_SAVE_CONFIG'
    KEY_SETUP_CREATE_SAMPLE = 'SETUP_CREATE_SAMPLE'
    KEY_SETUP_BASE_MODEL = 'SETUP_BASE_MODEL'
    KEY_SETUP_EPOCHS = 'SETUP_EPOCHS'
    KEY_SETUP_BATCH_SIZE = 'SETUP_BATCH_SIZE'
    KEY_SETUP_LR = 'SETUP_LR'
    KEY_SETUP_DELIMITER = 'SETUP_DELIMITER'
    KEY_SETUP_IN_PREFIX = 'SETUP_IN_PREFIX'
    KEY_SETUP_OUT_PREFIX = 'SETUP_OUT_PREFIX'

    # Process Tab Keys
    KEY_PROCESS_ADD_FILES = 'PROCESS_ADD_FILES'
    KEY_PROCESS_FILES_LIST = 'PROCESS_FILES_LIST'
    KEY_PROCESS_REMOVE_FILE = 'PROCESS_REMOVE_FILE'
    KEY_PROCESS_CLEAR_FILES = 'PROCESS_CLEAR_FILES'
    KEY_PROCESS_DELIMITER = 'PROCESS_DELIMITER'
    KEY_PROCESS_IN_PREFIX = 'PROCESS_IN_PREFIX'
    KEY_PROCESS_OUT_PREFIX = 'PROCESS_OUT_PREFIX'
    KEY_PROCESS_OUTPUT_FILE = 'PROCESS_OUTPUT_FILE'
    KEY_PROCESS_BROWSE_OUTPUT = 'PROCESS_BROWSE_OUTPUT'
    KEY_PROCESS_PREVIEW = 'PROCESS_PREVIEW'
    KEY_PROCESS_LOAD_AFTER = 'PROCESS_LOAD_AFTER'
    KEY_PROCESS_START = 'PROCESS_START'
    KEY_PROCESS_OUTPUT_LOG = 'PROCESS_OUTPUT_LOG'

    # Train Tab Keys
    KEY_TRAIN_DATA_PATH = 'TRAIN_DATA_PATH'
    KEY_TRAIN_BROWSE_DATA = 'TRAIN_BROWSE_DATA'
    KEY_TRAIN_LOAD_DATA = 'TRAIN_LOAD_DATA'
    KEY_TRAIN_DATA_STATUS = 'TRAIN_DATA_STATUS'
    KEY_TRAIN_PREVIEW_DATA = 'TRAIN_PREVIEW_DATA'
    KEY_TRAIN_BASE_MODEL = 'TRAIN_BASE_MODEL'
    KEY_TRAIN_DISPLAY_NAME = 'TRAIN_DISPLAY_NAME'
    KEY_TRAIN_EPOCHS = 'TRAIN_EPOCHS'
    KEY_TRAIN_BATCH_SIZE = 'TRAIN_BATCH_SIZE'
    KEY_TRAIN_LR = 'TRAIN_LR'
    KEY_TRAIN_START = 'TRAIN_START'
    KEY_TRAIN_OUTPUT_LOG = 'TRAIN_OUTPUT_LOG'

    # Manage Tab Keys
    KEY_MANAGE_REFRESH = 'MANAGE_REFRESH'
    KEY_MANAGE_TABLE = 'MANAGE_TABLE'
    KEY_MANAGE_DELETE = 'MANAGE_DELETE'
    KEY_MANAGE_TEST_SELECTED = 'MANAGE_TEST_SELECTED'
    KEY_MANAGE_CHAT_SELECTED = 'MANAGE_CHAT_SELECTED'
    KEY_MANAGE_STATUS_SELECTED = 'MANAGE_STATUS_SELECTED'
    KEY_MANAGE_OUTPUT_LOG = 'MANAGE_OUTPUT_LOG'

    # Test Tab Keys
    KEY_TEST_MODEL_SELECT = 'TEST_MODEL_SELECT'
    KEY_TEST_REFRESH_MODELS = 'TEST_REFRESH_MODELS'
    KEY_TEST_INPUT = 'TEST_INPUT'
    KEY_TEST_LOAD_INPUT = 'TEST_LOAD_INPUT'
    KEY_TEST_TEMP = 'TEST_TEMP'
    KEY_TEST_MAX_TOKENS = 'TEST_MAX_TOKENS'
    KEY_TEST_START = 'TEST_START'
    KEY_TEST_OUTPUT = 'TEST_OUTPUT'
    KEY_TEST_SAVE_OUTPUT = 'TEST_SAVE_OUTPUT'
    KEY_TEST_OUTPUT_LOG = 'TEST_OUTPUT_LOG'

    # Chat Tab Keys
    KEY_CHAT_MODEL_SELECT = 'CHAT_MODEL_SELECT'
    KEY_CHAT_REFRESH_MODELS = 'CHAT_REFRESH_MODELS'
    KEY_CHAT_START_SESSION = 'CHAT_START_SESSION'
    KEY_CHAT_DISPLAY = 'CHAT_DISPLAY'
    KEY_CHAT_INPUT = 'CHAT_INPUT'
    KEY_CHAT_SEND = 'CHAT_SEND'
    KEY_CHAT_END_SESSION = 'CHAT_END_SESSION'
    KEY_CHAT_CLEAR = 'CHAT_CLEAR'
    KEY_CHAT_STATUS = 'CHAT_STATUS'

    # Thread Events
    EVENT_THREAD_DONE = '-THREAD_DONE-'
    EVENT_THREAD_STATUS = '-THREAD_STATUS-'
    EVENT_THREAD_ERROR = '-THREAD_ERROR-'
    EVENT_THREAD_RESULT = '-THREAD_RESULT-' # Specific result payload
    EVENT_CHAT_CHUNK = '-CHAT_CHUNK-'


    def __init__(self, config: Config, processor: DataProcessor, model_manager: ModelManager):
        self.config = config
        self.processor = processor
        self.model_manager = model_manager
        self.window: Optional[sg.Window] = None
        self.training_data: List[Dict[str, str]] = []
        self.model_list_cache: List[Dict[str, Any]] = [] # Cache for model dropdowns
        self.active_chat_id: Optional[str] = None
        self.active_chat_model: Optional[str] = None

        if HAS_GUI:
            # sg.theme('DarkBlue3') # Example theme
            sg.theme('SystemDefaultForReal') # Try to match system
            sg.set_options(logging_level=logging.WARN) # Reduce PySimpleGUI's own logging
        else:
             logger.error("PySimpleGUI not found. GUI cannot run.")

    def run(self) -> None:
        """Run the GUI application."""
        if not HAS_GUI:
            print("Cannot start GUI: PySimpleGUI is not installed.")
            print("Install it using: pip install PySimpleGUI")
            return

        self.window = self._create_main_window()
        self._update_api_status_display() # Initial check
        self._refresh_model_lists() # Initial model list fetch

        # --- Main Event Loop ---
        while True:
            try:
                event, values = self.window.read()
                # print(f"Event: {event}, Values: {values}") # Debugging

                if event == sg.WINDOW_CLOSED:
                    break

                # --- Navigation ---
                if event.startswith('NAV_'):
                    self._handle_navigation(event)

                # --- Setup Tab ---
                elif event == self.KEY_SETUP_TEST_API:
                    self._run_in_thread(self._validate_api_key_thread, values[self.KEY_SETUP_API_KEY])
                elif event == self.KEY_SETUP_SAVE_CONFIG:
                    self._save_gui_config(values)
                elif event == self.KEY_SETUP_CREATE_SAMPLE:
                     self._create_sample_config_gui()

                # --- Process Tab ---
                elif event == self.KEY_PROCESS_ADD_FILES:
                     self._add_process_files(values)
                elif event == self.KEY_PROCESS_REMOVE_FILE:
                     self._remove_process_files(values)
                elif event == self.KEY_PROCESS_CLEAR_FILES:
                     self.window[self.KEY_PROCESS_FILES_LIST].update([])
                elif event == self.KEY_PROCESS_START:
                     self._run_in_thread(self._process_files_thread, values)
                elif event == self.KEY_PROCESS_BROWSE_OUTPUT:
                     # Let the FileSaveAs element handle this, value updated automatically
                     pass

                # --- Train Tab ---
                elif event == self.KEY_TRAIN_BROWSE_DATA:
                     pass # FileBrowse element handles this
                elif event == self.KEY_TRAIN_LOAD_DATA:
                     self._run_in_thread(self._load_training_data_thread, values[self.KEY_TRAIN_DATA_PATH])
                elif event == self.KEY_TRAIN_PREVIEW_DATA:
                     self._preview_gui_data()
                elif event == self.KEY_TRAIN_START:
                     self._run_in_thread(self._train_model_thread, values)

                # --- Manage Tab ---
                elif event == self.KEY_MANAGE_REFRESH:
                     self._run_in_thread(self._refresh_model_list_thread)
                elif event == self.KEY_MANAGE_DELETE:
                     self._delete_selected_model(values)
                elif event == self.KEY_MANAGE_TEST_SELECTED:
                      self._jump_to_test_with_selected(values)
                elif event == self.KEY_MANAGE_CHAT_SELECTED:
                      self._jump_to_chat_with_selected(values)
                elif event == self.KEY_MANAGE_STATUS_SELECTED:
                      self._check_selected_model_status(values)

                # --- Test Tab ---
                elif event == self.KEY_TEST_REFRESH_MODELS:
                     self._run_in_thread(self._refresh_model_list_thread) # Reuse refresh thread
                elif event == self.KEY_TEST_LOAD_INPUT:
                     self._load_test_input(values)
                elif event == self.KEY_TEST_START:
                     self._run_in_thread(self._test_model_thread, values)
                elif event == self.KEY_TEST_SAVE_OUTPUT:
                     self._save_test_output(values)

                # --- Chat Tab ---
                elif event == self.KEY_CHAT_REFRESH_MODELS:
                    self._run_in_thread(self._refresh_model_list_thread)
                elif event == self.KEY_CHAT_START_SESSION:
                    self._start_chat_session(values)
                elif event == self.KEY_CHAT_SEND or (event == self.KEY_CHAT_INPUT and values[self.KEY_CHAT_INPUT].endswith('\n')):
                    self._send_chat_message(values)
                elif event == self.KEY_CHAT_END_SESSION:
                    self._end_chat_session()
                elif event == self.KEY_CHAT_CLEAR:
                    self.window[self.KEY_CHAT_DISPLAY].update('')

                # --- Thread Communication Events ---
                elif event == self.EVENT_THREAD_STATUS:
                     self._update_status(values[event])
                     # Update specific log areas if applicable
                     if 'log_key' in values:
                          self.window[values['log_key']].print(values[event])
                elif event == self.EVENT_THREAD_ERROR:
                     error_msg = values[event]
                     self._update_status(f"Error: {error_msg}", error=True)
                     sg.popup_error(f"An error occurred:\n\n{error_msg}", title="Error")
                     # Update specific log areas if applicable
                     if 'log_key' in values:
                         self.window[values['log_key']].print(f"ERROR: {error_msg}")
                elif event == self.EVENT_THREAD_DONE:
                     origin_key = values.get('origin')
                     self._update_status(f"{origin_key or 'Task'} finished.", success=True)
                     # Re-enable buttons etc. based on origin if needed
                elif event == self.EVENT_THREAD_RESULT:
                     # Handle specific results from threads
                     payload = values[event]
                     origin = payload.get('origin')
                     data = payload.get('data')
                     if origin == '_validate_api_key_thread':
                          self._update_api_status_display(is_valid=data)
                     elif origin == '_refresh_model_list_thread':
                          self._update_model_dropdowns(data)
                          self._update_manage_table(data)
                     elif origin == '_load_training_data_thread':
                          self.training_data = data # Store loaded data
                          self.window[self.KEY_TRAIN_DATA_STATUS].update(f"{len(data)} examples loaded.")
                     elif origin == '_process_files_thread':
                          self._handle_process_result(data, values) # data = (processed_data, valid, invalid, output_file)
                     elif origin == '_train_model_thread':
                          job_ref = data # data = submitted_job_ref
                          self.window[self.KEY_TRAIN_OUTPUT_LOG].print(f"Training job submitted successfully! Display Name: {job_ref}")
                          self.window[self.KEY_TRAIN_OUTPUT_LOG].print("Check status using the Model Manager tab.")
                          sg.popup("Training Submitted", f"Job '{job_ref}' submitted.\nCheck Model Manager for status updates.")
                     elif origin == '_test_model_thread':
                         self.window[self.KEY_TEST_OUTPUT].update(data or "[No Response / Error]")
                         self.window[self.KEY_TEST_START].update(disabled=False)
                         self.window[self.KEY_TEST_OUTPUT_LOG].print(f"--- Response ---\n{data or '[No Response / Error]'}\n--------------")
                     elif origin == '_check_selected_model_status_thread':
                         self._display_model_status_popup(data)

                elif event == self.EVENT_CHAT_CHUNK:
                     # Append chunk to chat display
                     self.window[self.KEY_CHAT_DISPLAY].update(values[event], append=True)


            except Exception as e:
                 # Catch unexpected errors in the event loop
                 logger.error(f"Unexpected GUI Error: {e}", exc_info=True)
                 sg.popup_error(f"An unexpected error occurred in the application:\n\n{e}\n\nPlease check the console log for more details.", title="Application Error")
                 # Optionally break or try to recover
                 break


        # --- Cleanup ---
        if self.active_chat_id:
             self._end_chat_session() # Ensure chat is ended if window closed
        if self.window:
            self.window.close()
        logger.info("GUI closed.")

    # --- Window Creation ---

    def _create_main_window(self) -> sg.Window:
        """Creates the main application window with all tabs."""

        nav_buttons = [
            [sg.Button('Setup', key=self.KEY_NAV_SETUP, size=(12, 1))],
            [sg.Button('Process Files', key=self.KEY_NAV_PROCESS, size=(12, 1))],
            [sg.Button('Train Model', key=self.KEY_NAV_TRAIN, size=(12, 1))],
            [sg.Button('Test Model', key=self.KEY_NAV_TEST, size=(12, 1))],
            [sg.Button('Manage Models', key=self.KEY_NAV_MANAGE, size=(12, 1))],
            [sg.Button('Chat', key=self.KEY_NAV_CHAT, size=(12, 1))],
        ]

        navigation_col = sg.Column(nav_buttons, pad=(10, 10), vertical_alignment='top')

        content_area = sg.Column([
            [sg.Column(self._create_setup_layout(), key=self.KEY_CONTENT_SETUP, visible=True, expand_x=True, expand_y=True)],
            [sg.Column(self._create_process_layout(), key=self.KEY_CONTENT_PROCESS, visible=False, expand_x=True, expand_y=True)],
            [sg.Column(self._create_train_layout(), key=self.KEY_CONTENT_TRAIN, visible=False, expand_x=True, expand_y=True)],
            [sg.Column(self._create_test_layout(), key=self.KEY_CONTENT_TEST, visible=False, expand_x=True, expand_y=True)],
            [sg.Column(self._create_manage_layout(), key=self.KEY_CONTENT_MANAGE, visible=False, expand_x=True, expand_y=True)],
             [sg.Column(self._create_chat_layout(), key=self.KEY_CONTENT_CHAT, visible=False, expand_x=True, expand_y=True)],
        ], vertical_alignment='top', expand_x=True, expand_y=True, pad=(0,0))

        status_bar = sg.StatusBar("Ready", key=self.KEY_STATUS, size=(80, 1), justification='left')

        layout = [
            [sg.Text("🔮 Gemini Tuner", font=('Helvetica', 18), pad=((0,0),(5,10)))],
            [sg.HorizontalSeparator()],
            [navigation_col, sg.VerticalSeparator(), content_area],
            [sg.HorizontalSeparator()],
            [status_bar]
        ]

        window = sg.Window("Gemini Tuner", layout, resizable=True, finalize=True, size=(900, 650))
        # Make chat input handle Enter key
        window[self.KEY_CHAT_INPUT].bind('<Return>', '_Enter')
        return window

    def _create_setup_layout(self) -> List[List[sg.Element]]:
        api_key_frame = sg.Frame("API Key", [
            [sg.Text("Google AI API Key:", size=(18,1)), sg.Input(self.config.get('api_key', ''), key=self.KEY_SETUP_API_KEY, password_char='*'),
             sg.Button("Test", key=self.KEY_SETUP_TEST_API)],
            [sg.Text("Status: Unknown", key=self.KEY_SETUP_API_STATUS, size=(40,1))],
            [sg.Button("Save Configuration", key=self.KEY_SETUP_SAVE_CONFIG), sg.Button("Create Sample .env", key=self.KEY_SETUP_CREATE_SAMPLE)],
            [sg.Text("API Key required for all model operations (training, testing, listing, chat).", font="Helvetica 9")]
        ], expand_x=True)

        tuning_defaults_frame = sg.Frame("Default Tuning Settings", [
             [sg.Text("Base Model:", size=(18,1)), sg.Combo(self.model_manager.TUNABLE_MODELS, default_value=self.config.get('base_model'), key=self.KEY_SETUP_BASE_MODEL, size=(30,1))],
             [sg.Text("Epochs:", size=(18,1)), sg.Input(self.config.get('epochs'), key=self.KEY_SETUP_EPOCHS, size=(5,1))],
             [sg.Text("Batch Size:", size=(18,1)), sg.Input(self.config.get('batch_size'), key=self.KEY_SETUP_BATCH_SIZE, size=(5,1))],
             [sg.Text("Learning Rate Multiplier:", size=(18,1)), sg.Input(self.config.get('learning_rate_multiplier'), key=self.KEY_SETUP_LR, size=(5,1))],
        ], expand_x=True)

        format_defaults_frame = sg.Frame("Default File Format Settings", [
             [sg.Text("Example Delimiter:", size=(18,1)), sg.Input(self.config.get('delimiter'), key=self.KEY_SETUP_DELIMITER)],
             [sg.Text("Input Prefix:", size=(18,1)), sg.Input(self.config.get('input_prefix'), key=self.KEY_SETUP_IN_PREFIX)],
             [sg.Text("Output Prefix:", size=(18,1)), sg.Input(self.config.get('output_prefix'), key=self.KEY_SETUP_OUT_PREFIX)],
        ], expand_x=True)

        return [
            [sg.Text("Configuration", font=('Helvetica', 16))],
            [api_key_frame],
            [tuning_defaults_frame],
            [format_defaults_frame]
        ]

    def _create_process_layout(self) -> List[List[sg.Element]]:
         input_frame = sg.Frame("Input Files (.txt)", [
              [sg.Button("Add Files", key=self.KEY_PROCESS_ADD_FILES), sg.Button("Clear List", key=self.KEY_PROCESS_CLEAR_FILES)],
              [sg.Listbox([], size=(60, 6), key=self.KEY_PROCESS_FILES_LIST, enable_events=True, select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED)],
              [sg.Button("Remove Selected", key=self.KEY_PROCESS_REMOVE_FILE)],
         ], expand_x=True)

         format_frame = sg.Frame("Format Settings (Overrides Defaults)", [
              [sg.Text("Example Delimiter:", size=(15,1)), sg.Input(self.config.get('delimiter'), key=self.KEY_PROCESS_DELIMITER, size=(20,1))],
              [sg.Text("Input Prefix:", size=(15,1)), sg.Input(self.config.get('input_prefix'), key=self.KEY_PROCESS_IN_PREFIX, size=(20,1))],
              [sg.Text("Output Prefix:", size=(15,1)), sg.Input(self.config.get('output_prefix'), key=self.KEY_PROCESS_OUT_PREFIX, size=(20,1))],
         ], expand_x=True)

         output_frame = sg.Frame("Output", [
              [sg.Text("Save As (.jsonl):", size=(15,1)), sg.Input(f"training_data_{datetime.now().strftime('%Y%m%d')}.jsonl", key=self.KEY_PROCESS_OUTPUT_FILE, size=(30,1)),
               sg.FileSaveAs("Browse", key=self.KEY_PROCESS_BROWSE_OUTPUT, file_types=(("JSON Lines", "*.jsonl"),))],
              [sg.Checkbox("Preview first 3 examples after processing", default=True, key=self.KEY_PROCESS_PREVIEW)],
              [sg.Checkbox("Load data for training after saving", default=True, key=self.KEY_PROCESS_LOAD_AFTER)],
              [sg.Button("Start Processing", key=self.KEY_PROCESS_START, button_color=('white', 'green'))],
         ], expand_x=True)

         log_frame = sg.Frame("Processing Log", [[
             sg.Multiline(size=(80, 8), key=self.KEY_PROCESS_OUTPUT_LOG, autoscroll=True, reroute_stdout=False, reroute_stderr=False, disabled=True) # Use print(..., file=...) later
         ]], expand_x=True)


         return [
            [sg.Text("Process Text Files into Training Data", font=('Helvetica', 16))],
            [input_frame],
            [format_frame],
            [output_frame],
            [log_frame]
         ]

    def _create_train_layout(self) -> List[List[sg.Element]]:
         data_frame = sg.Frame("Training Data (.jsonl)", [
              [sg.Text("Data File:", size=(10,1)), sg.Input(key=self.KEY_TRAIN_DATA_PATH, size=(40,1)),
               sg.FileBrowse("Browse", key=self.KEY_TRAIN_BROWSE_DATA, file_types=(("JSON Lines", "*.jsonl"),)),
               sg.Button("Load Data", key=self.KEY_TRAIN_LOAD_DATA)],
              [sg.Text("Status: No data loaded", key=self.KEY_TRAIN_DATA_STATUS, size=(40,1)),
               sg.Button("Preview Data", key=self.KEY_TRAIN_PREVIEW_DATA, disabled=True)],
         ], expand_x=True)

         settings_frame = sg.Frame("Tuning Settings", [
              [sg.Text("Base Model:", size=(15,1)), sg.Combo(self.model_manager.TUNABLE_MODELS, default_value=self.config.get('base_model'), key=self.KEY_TRAIN_BASE_MODEL, size=(30,1))],
              [sg.Text("Display Name:", size=(15,1)), sg.Input(f"Tuned_{Path(self.config.get('base_model')).name}_{datetime.now().strftime('%Y%m%d_%H%M')}", key=self.KEY_TRAIN_DISPLAY_NAME, size=(30,1))],
              [sg.Text("Epochs:", size=(15,1)), sg.Input(self.config.get('epochs'), key=self.KEY_TRAIN_EPOCHS, size=(5,1))],
              [sg.Text("Batch Size:", size=(15,1)), sg.Input(self.config.get('batch_size'), key=self.KEY_TRAIN_BATCH_SIZE, size=(5,1))],
              [sg.Text("Learning Rate Multiplier:", size=(15,1)), sg.Input(self.config.get('learning_rate_multiplier'), key=self.KEY_TRAIN_LR, size=(5,1))],
         ], expand_x=True)

         action_frame = sg.Frame("Action", [
              [sg.Button("Start Training Job", key=self.KEY_TRAIN_START, button_color=('white', 'green'), disabled=True)],
              [sg.Text("Training runs in the background on Google Cloud. Check Model Manager for status.", font="Helvetica 9")]
         ], expand_x=True)

         log_frame = sg.Frame("Training Log", [[
             sg.Multiline(size=(80, 8), key=self.KEY_TRAIN_OUTPUT_LOG, autoscroll=True, reroute_stdout=False, reroute_stderr=False, disabled=True)
         ]], expand_x=True)


         return [
             [sg.Text("Train a New Model", font=('Helvetica', 16))],
             [data_frame],
             [settings_frame],
             [action_frame],
             [log_frame]
         ]

    def _create_manage_layout(self) -> List[List[sg.Element]]:
         table_headings = ["Display Name", "State", "Base Model", "Created", "Model ID"]
         col_widths = [25, 10, 20, 16, 30] # Adjust as needed

         table_frame = sg.Frame("Your Tuned Models", [
             [sg.Button("Refresh List", key=self.KEY_MANAGE_REFRESH)],
             [sg.Table(values=[], headings=table_headings, key=self.KEY_MANAGE_TABLE,
                       auto_size_columns=False, col_widths=col_widths, justification='left',
                       num_rows=10, enable_events=True, select_mode=sg.TABLE_SELECT_MODE_BROWSE,
                       expand_x=True)],
              [sg.Button("Delete Selected", key=self.KEY_MANAGE_DELETE, button_color=('white', 'red'), disabled=True),
               sg.Button("Check Status", key=self.KEY_MANAGE_STATUS_SELECTED, disabled=True),
               sg.Button("Test Selected", key=self.KEY_MANAGE_TEST_SELECTED, disabled=True),
               sg.Button("Chat with Selected", key=self.KEY_MANAGE_CHAT_SELECTED, disabled=True)
               ],
         ], expand_x=True)

         log_frame = sg.Frame("Manager Log", [[
             sg.Multiline(size=(80, 5), key=self.KEY_MANAGE_OUTPUT_LOG, autoscroll=True, reroute_stdout=False, reroute_stderr=False, disabled=True)
         ]], expand_x=True)

         return [
             [sg.Text("Manage Tuned Models", font=('Helvetica', 16))],
             [table_frame],
             [log_frame]
         ]

    def _create_test_layout(self) -> List[List[sg.Element]]:
         model_frame = sg.Frame("Select Model", [
             [sg.Text("Model:", size=(8,1)), sg.Combo([], key=self.KEY_TEST_MODEL_SELECT, size=(50,1), enable_events=True, readonly=True), # Populate dynamically
              sg.Button("Refresh", key=self.KEY_TEST_REFRESH_MODELS)],
             # Add details display here if needed
         ], expand_x=True)

         input_frame = sg.Frame("Input Prompt", [
             [sg.Button("Load from File", key=self.KEY_TEST_LOAD_INPUT)],
             [sg.Multiline(size=(80, 6), key=self.KEY_TEST_INPUT)],
         ], expand_x=True)

         config_frame = sg.Frame("Generation Config (Optional)", [
              [sg.Text("Temp:", size=(5,1)), sg.Input("0.7", key=self.KEY_TEST_TEMP, size=(5,1)),
               sg.Text("Max Tokens:", size=(10,1)), sg.Input("1024", key=self.KEY_TEST_MAX_TOKENS, size=(7,1))]
              # Could add top_p, top_k
         ])

         action_frame = sg.Frame("Action", [
              [sg.Button("Generate Response", key=self.KEY_TEST_START, button_color=('white', 'green'), disabled=True)]
         ])

         output_frame = sg.Frame("Output Response", [
             [sg.Multiline(size=(80, 10), key=self.KEY_TEST_OUTPUT, disabled=True)],
             [sg.Button("Save Output", key=self.KEY_TEST_SAVE_OUTPUT)]
         ], expand_x=True)

         log_frame = sg.Frame("Test Log", [[
             sg.Multiline(size=(80, 4), key=self.KEY_TEST_OUTPUT_LOG, autoscroll=True, reroute_stdout=False, reroute_stderr=False, disabled=True)
         ]], expand_x=True)


         return [
             [sg.Text("Test a Model", font=('Helvetica', 16))],
             [model_frame],
             [input_frame],
             [config_frame, action_frame], # Place side-by-side
             [output_frame],
             [log_frame]
         ]

    def _create_chat_layout(self) -> List[List[sg.Element]]:
        model_select_frame = sg.Frame("Select Model for Chat", [
            [sg.Combo([], key=self.KEY_CHAT_MODEL_SELECT, size=(50, 1), enable_events=True, readonly=True),
             sg.Button("Refresh Models", key=self.KEY_CHAT_REFRESH_MODELS),
             sg.Button("Start Chat", key=self.KEY_CHAT_START_SESSION, disabled=True)]
        ], expand_x=True)

        chat_display_frame = sg.Frame("Conversation", [
            [sg.Multiline(key=self.KEY_CHAT_DISPLAY, size=(80, 15), autoscroll=True, disabled=True, reroute_cprint=True)] # Use cprint later
        ], expand_x=True, expand_y=True)

        chat_input_frame = sg.Frame("Send Message", [
            [sg.Multiline(key=self.KEY_CHAT_INPUT, size=(70, 3), do_not_clear=False, enable_events=True), # Enter key event bound in main loop
             sg.Button("Send", key=self.KEY_CHAT_SEND, bind_return_key=True, disabled=True)],
             [sg.Button("End Chat", key=self.KEY_CHAT_END_SESSION, button_color=('white','orange'), disabled=True), sg.Button("Clear Display", key=self.KEY_CHAT_CLEAR)]
        ], expand_x=True)

        status_frame = sg.Frame("Chat Status", [
             [sg.Text("Status: Idle", key=self.KEY_CHAT_STATUS, size=(80,1))]
        ], expand_x=True)


        return [
            [sg.Text("Chat with Gemini", font=('Helvetica', 16))],
            [model_select_frame],
            [chat_display_frame],
            [chat_input_frame],
            [status_frame]
        ]


    # --- GUI Event Handlers & Logic ---

    def _handle_navigation(self, event: str) -> None:
        """Switch visible content based on navigation button press."""
        content_keys = [
            self.KEY_CONTENT_SETUP, self.KEY_CONTENT_PROCESS, self.KEY_CONTENT_TRAIN,
            self.KEY_CONTENT_TEST, self.KEY_CONTENT_MANAGE, self.KEY_CONTENT_CHAT
        ]
        target_content = f"CONTENT_{event[4:]}" # e.g., NAV_SETUP -> CONTENT_SETUP

        for key in content_keys:
            self.window[key].update(visible=(key == target_content))

        # Update status bar
        nav_name = event[4:].replace('_', ' ')
        self._update_status(f"Switched to {nav_name} tab.")

        # Actions on tab switch
        if event == self.KEY_NAV_SETUP:
             # Refresh API status display in case key was changed externally
             self._update_api_status_display()
             # Populate config fields from current config
             self._populate_setup_fields()
        elif event == self.KEY_NAV_MANAGE:
             self._run_in_thread(self._refresh_model_list_thread) # Refresh model list when entering tab
        elif event in [self.KEY_NAV_TEST, self.KEY_NAV_CHAT]:
             # Refresh model dropdowns if cache is empty or stale (or always)
             if not self.model_list_cache:
                self._run_in_thread(self._refresh_model_list_thread)

    def _update_status(self, message: str, success: bool = False, error: bool = False) -> None:
        """Update the status bar text and color."""
        if not self.window: return
        color = 'black'
        if success: color = 'darkgreen'
        if error: color = 'red'
        # Ensure message is a string
        message_str = str(message) if message is not None else ""
        self.window[self.KEY_STATUS].update(message_str, text_color=color)
        # Optionally log to console as well
        if error:
            logger.error(f"Status Update: {message_str}")
        else:
            logger.info(f"Status Update: {message_str}")


    def _run_in_thread(self, target_func: callable, *args, origin: Optional[str] = None, **kwargs) -> None:
        """Run a function in a separate thread and notify the GUI."""
        if not self.window: return
        if not origin:
             origin = target_func.__name__ # Use function name as default origin

        # Disable relevant buttons here (e.g., the button that triggered the action)
        # Example: self.window[self.KEY_PROCESS_START].update(disabled=True)

        self._update_status(f"Running {origin}...")

        thread = threading.Thread(
            target=self._thread_wrapper,
            args=(target_func, args, kwargs, origin),
            daemon=True
        )
        thread.start()

    def _thread_wrapper(self, target_func: callable, args: tuple, kwargs: dict, origin: str):
        """Wrapper for thread execution to send results/errors back to GUI."""
        if not self.window: return
        log_key = None # Determine log key based on origin if possible
        if origin == '_process_files_thread': log_key = self.KEY_PROCESS_OUTPUT_LOG
        elif origin == '_train_model_thread': log_key = self.KEY_TRAIN_OUTPUT_LOG
        elif origin == '_test_model_thread': log_key = self.KEY_TEST_OUTPUT_LOG
        elif origin == '_refresh_model_list_thread': log_key = self.KEY_MANAGE_OUTPUT_LOG # Log in manage tab

        status_update = lambda msg: self.window.write_event_value(self.EVENT_THREAD_STATUS, {'value': msg, 'origin': origin, 'log_key': log_key})
        try:
            # Pass status_update callback to the target function if it accepts it
            if 'status_callback' in target_func.__code__.co_varnames:
                kwargs['status_callback'] = status_update

            result = target_func(*args, **kwargs)
            # Send result back
            self.window.write_event_value(self.EVENT_THREAD_RESULT, {'origin': origin, 'data': result})
            self.window.write_event_value(self.EVENT_THREAD_DONE, {'origin': origin}) # Signal completion
        except Exception as e:
            logger.error(f"Error in thread {origin}: {e}", exc_info=True)
            self.window.write_event_value(self.EVENT_THREAD_ERROR, {'value': str(e), 'origin': origin, 'log_key': log_key})
        finally:
            # Re-enable buttons maybe? Or do it when EVENT_THREAD_DONE is received
             pass


    # --- Setup Tab Methods ---
    def _populate_setup_fields(self):
         if not self.window: return
         self.window[self.KEY_SETUP_API_KEY].update(self.config.get('api_key', ''))
         self.window[self.KEY_SETUP_BASE_MODEL].update(value=self.config.get('base_model'))
         self.window[self.KEY_SETUP_EPOCHS].update(self.config.get('epochs'))
         self.window[self.KEY_SETUP_BATCH_SIZE].update(self.config.get('batch_size'))
         self.window[self.KEY_SETUP_LR].update(self.config.get('learning_rate_multiplier'))
         self.window[self.KEY_SETUP_DELIMITER].update(self.config.get('delimiter'))
         self.window[self.KEY_SETUP_IN_PREFIX].update(self.config.get('input_prefix'))
         self.window[self.KEY_SETUP_OUT_PREFIX].update(self.config.get('output_prefix'))


    def _validate_api_key_thread(self, api_key: str, status_callback: callable = None) -> bool:
        """Thread function to validate API key."""
        if not api_key:
             if status_callback: status_callback("No API key entered.")
             return False
        if status_callback: status_callback("Validating API key...")
        is_valid = self.model_manager.validate_api_key(api_key)
        if is_valid:
             if status_callback: status_callback("API Key is valid.")
             # Update config and re-initialize client *in main thread* via event? Or here?
             # Let's update config here, re-init can happen via status update handler
             self.config.update(api_key=api_key)
             self.model_manager._initialize_genai() # Re-init client
        else:
             if status_callback: status_callback("API Key is invalid or permissions insufficient.")
             # If the current client was using this invalid key, disable it
             if self.config.get('api_key') == api_key:
                  self.model_manager.client = None # Or re-init without key

        return is_valid

    def _update_api_status_display(self, is_valid: Optional[bool] = None) -> None:
        """Update the API Status text in the Setup tab."""
        if not self.window: return
        status_text = "Status: "
        color = "black"
        key_present = bool(self.config.get('api_key'))

        if is_valid is True:
            status_text += "Valid"
            color = "darkgreen"
        elif is_valid is False:
             status_text += "Invalid / Permission Denied"
             color = "red"
        elif key_present and self.model_manager.is_client_ready():
             # Already initialized and likely valid from start/previous check
             status_text += "Valid (Initialized)"
             color = "darkgreen"
        elif key_present:
             status_text += "Not verified (Press Test)"
             color = "orange"
        else:
             status_text += "Not Set"
             color = "red"

        self.window[self.KEY_SETUP_API_STATUS].update(status_text, text_color=color)
        # Enable/disable relevant features based on API status
        api_ready = self.model_manager.is_client_ready()
        self.window[self.KEY_NAV_TRAIN].update(disabled=not api_ready)
        self.window[self.KEY_NAV_TEST].update(disabled=not api_ready)
        self.window[self.KEY_NAV_MANAGE].update(disabled=not api_ready)
        self.window[self.KEY_NAV_CHAT].update(disabled=not api_ready)
        # Also update buttons within tabs if needed (e.g., Start Training)
        self.window[self.KEY_TRAIN_START].update(disabled=not (api_ready and self.training_data))
        self.window[self.KEY_TEST_START].update(disabled=not api_ready)
        # Manage buttons need selection AND api_ready
        self.window[self.KEY_MANAGE_DELETE].update(disabled=not api_ready)
        self.window[self.KEY_MANAGE_STATUS_SELECTED].update(disabled=not api_ready)
        self.window[self.KEY_MANAGE_TEST_SELECTED].update(disabled=not api_ready)
        self.window[self.KEY_MANAGE_CHAT_SELECTED].update(disabled=not api_ready)
        self.window[self.KEY_CHAT_START_SESSION].update(disabled=not api_ready)


    def _save_gui_config(self, values: Dict) -> None:
        """Update config from GUI fields and save."""
        try:
            self.config.update(
                api_key=values[self.KEY_SETUP_API_KEY],
                base_model=values[self.KEY_SETUP_BASE_MODEL],
                epochs=values[self.KEY_SETUP_EPOCHS],
                batch_size=values[self.KEY_SETUP_BATCH_SIZE],
                learning_rate_multiplier=values[self.KEY_SETUP_LR],
                delimiter=values[self.KEY_SETUP_DELIMITER],
                input_prefix=values[self.KEY_SETUP_IN_PREFIX],
                output_prefix=values[self.KEY_SETUP_OUT_PREFIX]
            )
            self.config.save_config()
            self._update_status("Configuration saved successfully.", success=True)
            sg.popup("Configuration Saved", "Default settings updated and saved to:", self.config.config_file or ".env")
        except Exception as e:
            self._update_status(f"Error saving configuration: {e}", error=True)
            sg.popup_error(f"Error saving configuration:\n{e}")

    def _create_sample_config_gui(self):
        """Action for the 'Create Sample .env' button."""
        save_path = sg.popup_get_file("Save sample config as", default_path=".env.example", save_as=True, file_types=((".env Example", ".env.example"),("All Files", "*.*")))
        if save_path:
            try:
                Config.create_sample_config(save_path)
                self._update_status(f"Sample config created at {save_path}", success=True)
                sg.popup("Sample File Created", f"Sample configuration saved to:\n{save_path}\n\nRemember to rename it to '.env' and add your API key.", title="Success")
            except Exception as e:
                 self._update_status(f"Failed to create sample config: {e}", error=True)
                 sg.popup_error(f"Failed to create sample config:\n{e}")


    # --- Process Tab Methods ---
    def _add_process_files(self, values: Dict):
        """Handle adding files to the processing list."""
        file_paths_str = sg.popup_get_file(
            "Select Text Files",
            multiple_files=True,
            file_types=(("Text Files", "*.txt"),("All Files", "*.*"))
        )
        if file_paths_str:
            file_paths = file_paths_str.split(';') # PySimpleGUI uses semicolon separator
            current_list = self.window[self.KEY_PROCESS_FILES_LIST].get_list_values()
            new_files = [fp for fp in file_paths if fp not in current_list]
            updated_list = current_list + new_files
            self.window[self.KEY_PROCESS_FILES_LIST].update(updated_list)
            self._update_status(f"Added {len(new_files)} file(s). Total: {len(updated_list)}.")

    def _remove_process_files(self, values: Dict):
        """Remove selected files from the processing list."""
        selected_files = values[self.KEY_PROCESS_FILES_LIST]
        if not selected_files:
             sg.popup_notify("No files selected to remove.")
             return
        current_list = self.window[self.KEY_PROCESS_FILES_LIST].get_list_values()
        updated_list = [f for f in current_list if f not in selected_files]
        self.window[self.KEY_PROCESS_FILES_LIST].update(updated_list)
        self._update_status(f"Removed {len(selected_files)} file(s). Total: {len(updated_list)}.")


    def _process_files_thread(self, values: Dict, status_callback: callable):
        """Thread function for processing files."""
        file_paths = values[self.KEY_PROCESS_FILES_LIST]
        output_file = values[self.KEY_PROCESS_OUTPUT_FILE]
        delimiter = values[self.KEY_PROCESS_DELIMITER]
        input_prefix = values[self.KEY_PROCESS_IN_PREFIX]
        output_prefix = values[self.KEY_PROCESS_OUT_PREFIX]

        if not file_paths:
            raise ValueError("No input files selected.")
        if not output_file:
            raise ValueError("Output file path cannot be empty.")

        # Use status_callback for intermediate logging
        log = lambda msg: status_callback(msg) if status_callback else print(msg)
        self.window[self.KEY_PROCESS_OUTPUT_LOG].update('') # Clear log

        log(f"Starting processing for {len(file_paths)} file(s)...")
        log(f"Output will be saved to: {output_file}")

        processed_data, valid_count, invalid_count = self.processor.process_multiple_files(
            file_paths,
            delimiter=delimiter,
            input_prefix=input_prefix,
            output_prefix=output_prefix
            # Note: status_callback isn't easily passed down here, relies on logger
        )

        log(f"\nProcessing finished: {valid_count} valid examples, {invalid_count} skipped.")

        if not processed_data:
             log("No valid data processed. Output file will not be saved.")
             return ([], 0, 0, None) # Return empty result

        try:
            log(f"Saving data to {output_file}...")
            self.processor.save_training_data(processed_data, output_file)
            log("Data saved successfully.")
        except Exception as e:
            log(f"ERROR saving output file: {e}")
            raise # Re-raise to be caught by _thread_wrapper

        return (processed_data, valid_count, invalid_count, output_file)


    def _handle_process_result(self, result_data, form_values):
         """Update GUI after processing thread finishes."""
         processed_data, valid_count, invalid_count, output_file = result_data

         if output_file: # If data was saved
             self._update_status(f"Processing complete. {valid_count} examples saved to {output_file}", success=True)
             if form_values.get(self.KEY_PROCESS_PREVIEW):
                  self._preview_gui_data(data=processed_data, title_suffix=f"from {Path(output_file).name}")
             if form_values.get(self.KEY_PROCESS_LOAD_AFTER):
                 self.training_data = processed_data
                 # Update train tab display
                 self.window[self.KEY_TRAIN_DATA_PATH].update(output_file)
                 self.window[self.KEY_TRAIN_DATA_STATUS].update(f"{len(processed_data)} examples loaded from processing.")
                 self.window[self.KEY_TRAIN_PREVIEW_DATA].update(disabled=False)
                 self.window[self.KEY_TRAIN_START].update(disabled=not self.model_manager.is_client_ready())
                 sg.popup("Data Loaded", f"{len(processed_data)} processed examples loaded and ready for training.")
             else:
                 sg.popup("Processing Complete", f"{valid_count} examples saved to:\n{output_file}")
         elif valid_count == 0:
             self._update_status(f"Processing finished, but no valid examples found.", error=True)
             sg.popup_warning("Processing Finished", "No valid examples were extracted from the input files.")
         else:
             # Should have been an error during saving if output_file is None but data existed
             self._update_status("Processing finished, but failed to save output.", error=True)



    # --- Train Tab Methods ---
    def _load_training_data_thread(self, data_path: str, status_callback: callable):
         """Thread function to load training data."""
         if not data_path:
              raise ValueError("No data file path specified.")
         log = lambda msg: status_callback(msg) if status_callback else print(msg)
         log(f"Loading training data from: {data_path}")
         try:
              data = self.processor.load_training_data(data_path)
              log(f"Successfully loaded {len(data)} examples.")
              return data
         except FileNotFoundError:
              log(f"Error: File not found at {data_path}")
              raise
         except Exception as e:
              log(f"Error loading data: {e}")
              raise

    def _preview_gui_data(self, data: Optional[List] = None, title_suffix=""):
         """Show a preview of training data in a popup."""
         data_to_show = data if data is not None else self.training_data
         if not data_to_show:
              sg.popup_notify("No data loaded to preview.")
              return

         num_to_show = min(15, len(data_to_show))
         preview_values = []
         for i, ex in enumerate(data_to_show[:num_to_show]):
             inp = str(ex.get('text_input', ''))
             outp = str(ex.get('output', ''))
             preview_values.append([
                 i + 1,
                 inp[:80] + ('...' if len(inp) > 80 else ''),
                 outp[:80] + ('...' if len(outp) > 80 else '')
             ])

         layout = [
             [sg.Text(f"Training Data Preview ({num_to_show} of {len(data_to_show)}) {title_suffix}", font=('Helvetica', 14))],
             [sg.Table(values=preview_values, headings=["#", "Input (preview)", "Output (preview)"],
                       auto_size_columns=False, col_widths=[5, 40, 40], justification='left',
                       num_rows=num_to_show, expand_x=True)],
             [sg.Button("Close")]
         ]
         preview_window = sg.Window("Data Preview", layout, modal=True, finalize=True)
         preview_window.read(close=True) # Wait for close


    def _train_model_thread(self, values: Dict, status_callback: callable):
         """Thread function to start model training."""
         if not self.training_data:
              raise ValueError("No training data loaded.")
         if not self.model_manager.is_client_ready():
              raise ConnectionError("API client not ready. Cannot train.")

         log = lambda msg: status_callback(msg) if status_callback else print(msg)
         self.window[self.KEY_TRAIN_OUTPUT_LOG].update('') # Clear log

         # Get params from form
         base_model = values[self.KEY_TRAIN_BASE_MODEL]
         display_name = values[self.KEY_TRAIN_DISPLAY_NAME]
         epochs = int(values[self.KEY_TRAIN_EPOCHS]) # Add validation
         batch_size = int(values[self.KEY_TRAIN_BATCH_SIZE]) # Add validation
         lr_multiplier = float(values[self.KEY_TRAIN_LR]) # Add validation

         log(f"Preparing to submit training job '{display_name}'...")
         log(f"Base Model: {base_model}, Epochs: {epochs}, Batch: {batch_size}, LR Multiplier: {lr_multiplier}")
         log(f"Using {len(self.training_data)} training examples.")

         if len(self.training_data) < 10:
             # Consider adding a confirmation popup here via main thread if really needed
             log("WARNING: Dataset size is very small (<10). Training might fail.")


         submitted_job_ref = self.model_manager.train_model(
             training_data=self.training_data,
             base_model=base_model,
             display_name=display_name,
             epochs=epochs,
             batch_size=batch_size,
             learning_rate_multiplier=lr_multiplier
         )

         if submitted_job_ref:
              log(f"Training job submitted. Display Name: {submitted_job_ref}")
              return submitted_job_ref # Return the display name (or operation name if available)
         else:
              log("ERROR: Failed to submit training job.")
              raise RuntimeError("Failed to submit training job. Check logs.")

    # --- Manage Tab Methods ---
    def _refresh_model_list_thread(self, status_callback: callable) -> List[Dict]:
         """Thread function to fetch the list of tuned models."""
         log = lambda msg: status_callback(msg) if status_callback else print(msg)
         if not self.model_manager.is_client_ready():
              log("API client not ready. Cannot list models.")
              return []
         log("Fetching list of tuned models...")
         models = self.model_manager.list_models(only_tuned=True)
         log(f"Found {len(models)} tuned models.")
         self.model_list_cache = models # Update cache
         return models

    def _update_manage_table(self, models: List[Dict]):
         """Update the table in the Manage Models tab."""
         if not self.window: return
         table_data = []
         for m in models:
             create_time = m.get('create_time', '')
             if hasattr(create_time, 'strftime'): create_time = create_time.strftime('%Y-%m-%d %H:%M')
             table_data.append([
                 m.get('display_name', 'N/A'),
                 m.get('state', 'N/A').upper(),
                 m.get('base_model', 'N/A'),
                 create_time,
                 m.get('name', 'N/A') # Model ID
             ])
         self.window[self.KEY_MANAGE_TABLE].update(values=table_data)
         # Disable action buttons initially
         self.window[self.KEY_MANAGE_DELETE].update(disabled=True)
         self.window[self.KEY_MANAGE_STATUS_SELECTED].update(disabled=True)
         self.window[self.KEY_MANAGE_TEST_SELECTED].update(disabled=True)
         self.window[self.KEY_MANAGE_CHAT_SELECTED].update(disabled=True)


    def _handle_manage_table_selection(self, values: Dict):
        """Enable/disable buttons based on table selection and model state."""
        if not self.window or not self.model_manager.is_client_ready(): return

        selected_indices = values[self.KEY_MANAGE_TABLE]
        can_delete = False
        can_check_status = False
        can_test = False
        can_chat = False

        if selected_indices:
            idx = selected_indices[0]
            if 0 <= idx < len(self.model_list_cache):
                model = self.model_list_cache[idx]
                state = model.get('state', '').upper()
                # Can delete any tuned model (unless maybe CREATING?)
                can_delete = True
                can_check_status = True # Can always check status
                # Can only test/chat with ACTIVE models
                if state == 'ACTIVE':
                    can_test = True
                    can_chat = True

        self.window[self.KEY_MANAGE_DELETE].update(disabled=not can_delete)
        self.window[self.KEY_MANAGE_STATUS_SELECTED].update(disabled=not can_check_status)
        self.window[self.KEY_MANAGE_TEST_SELECTED].update(disabled=not can_test)
        self.window[self.KEY_MANAGE_CHAT_SELECTED].update(disabled=not can_chat)


    def _get_selected_model_from_manage_table(self, values: Dict) -> Optional[Dict]:
         """Helper to get the full data dict for the selected model."""
         selected_indices = values.get(self.KEY_MANAGE_TABLE)
         if selected_indices:
             idx = selected_indices[0]
             if 0 <= idx < len(self.model_list_cache):
                  return self.model_list_cache[idx]
         return None

    def _delete_selected_model(self, values: Dict):
         """Handle deleting the selected model after confirmation."""
         model = self._get_selected_model_from_manage_table(values)
         if not model:
              sg.popup_notify("No model selected.")
              return

         model_name = model.get('name')
         display_name = model.get('display_name')

         confirm = sg.popup_yes_no(f"Are you sure you want to permanently delete this tuned model?\n\n"
                                  f"Name: {display_name}\n"
                                  f"ID: {model_name}",
                                  title="Confirm Deletion", button_color=('white','red'))

         if confirm == 'Yes':
              self._run_in_thread(self._delete_model_thread, model_name)

    def _delete_model_thread(self, model_name: str, status_callback: callable):
         """Thread function to delete a model."""
         log = lambda msg: status_callback(msg) if status_callback else print(msg)
         self.window[self.KEY_MANAGE_OUTPUT_LOG].update('') # Clear log
         log(f"Attempting to delete model: {model_name}")
         success = self.model_manager.delete_model(model_name)
         if success:
              log(f"Model {model_name} deleted successfully (or was not found).")
              # Trigger refresh in main thread
              self.window.write_event_value('-REFRESH_MODELS_AFTER_DELETE-', '') # Custom event
         else:
              log(f"Failed to delete model {model_name}. Check API permissions.")
              raise RuntimeError(f"Deletion failed for {model_name}")

    def _check_selected_model_status(self, values: Dict):
         model = self._get_selected_model_from_manage_table(values)
         if model and model.get('name'):
              self._run_in_thread(self._check_selected_model_status_thread, model.get('name'))
         else:
              sg.popup_notify("No model selected or model has no ID.")

    def _check_selected_model_status_thread(self, model_name: str, status_callback: callable) -> Optional[Dict]:
         log = lambda msg: status_callback(msg) if status_callback else print(msg)
         log(f"Checking status for: {model_name}")
         status = self.model_manager.get_tuning_job_status(model_name)
         if status:
             log(f"Status retrieved for {model_name}: {status.get('state')}")
             return status
         else:
             log(f"Could not retrieve status for {model_name}.")
             return None # Indicate failure

    def _display_model_status_popup(self, status: Optional[Dict]):
         if not status:
             sg.popup_error("Could not retrieve model status.", title="Status Check Failed")
             return

         create_time = status.get('create_time', '')
         update_time = status.get('update_time', '')
         if hasattr(create_time, 'strftime'): create_time = create_time.strftime('%Y-%m-%d %H:%M:%S')
         if hasattr(update_time, 'strftime'): update_time = update_time.strftime('%Y-%m-%d %H:%M:%S')

         message = (
             f"Model: {status.get('display_name', 'N/A')} ({status.get('name', 'N/A')})\n"
             f"State: {status.get('state', 'UNKNOWN').upper()}\n"
             f"Created: {create_time}\n"
             f"Updated: {update_time}\n"
         )
         sg.popup(message, title="Model Status")


    def _jump_to_test_with_selected(self, values: Dict):
        model = self._get_selected_model_from_manage_table(values)
        if model and model.get('name') and model.get('state','').upper() == 'ACTIVE':
            model_name = model.get('name')
            # Switch tab
            self._handle_navigation(self.KEY_NAV_TEST)
            # Set model in dropdown (ensure dropdown is populated first)
            self._update_model_dropdowns(self.model_list_cache) # Ensure it's up to date
            self.window[self.KEY_TEST_MODEL_SELECT].update(value=model_name)
            self._update_status(f"Switched to Test tab for model: {model.get('display_name')}")
        elif model and model.get('state','').upper() != 'ACTIVE':
             sg.popup_notify("Selected model is not ACTIVE. Cannot test.")
        else:
             sg.popup_notify("No active model selected.")

    def _jump_to_chat_with_selected(self, values: Dict):
        model = self._get_selected_model_from_manage_table(values)
        if model and model.get('name') and model.get('state','').upper() == 'ACTIVE':
            model_name = model.get('name')
            # Switch tab
            self._handle_navigation(self.KEY_NAV_CHAT)
            # Set model in dropdown
            self._update_model_dropdowns(self.model_list_cache)
            self.window[self.KEY_CHAT_MODEL_SELECT].update(value=model_name)
            self._update_status(f"Switched to Chat tab for model: {model.get('display_name')}")
            # Enable start chat button
            self.window[self.KEY_CHAT_START_SESSION].update(disabled=False)
        elif model and model.get('state','').upper() != 'ACTIVE':
             sg.popup_notify("Selected model is not ACTIVE. Cannot chat.")
        else:
             sg.popup_notify("No active model selected.")

    # --- Test Tab Methods ---
    def _update_model_dropdowns(self, models: List[Dict]):
         """Populate model selection dropdowns in Test and Chat tabs."""
         if not self.window: return
         # Include base models + active tuned models
         base_models = self.model_manager.get_tunable_models() # Or list all generative?
         active_tuned = [m for m in models if m.get('state', '').upper() == 'ACTIVE']

         display_map = {m:m for m in base_models} # Map ID to display name (same for base)
         for m in active_tuned:
             display_map[m['name']] = f"{m['display_name']} ({m['name']})" # Show name and ID

         # Sort keys (model IDs) - maybe base first, then tuned by name?
         sorted_ids = sorted(base_models) + sorted([m['name'] for m in active_tuned], key=lambda x: display_map[x])

         # Get display names in sorted order
         display_values = [display_map[id] for id in sorted_ids]

         # Store the mapping from display name back to ID
         self.test_model_display_to_id = {display: id for id, display in display_map.items()}

         # Update dropdowns
         current_test_val = self.window[self.KEY_TEST_MODEL_SELECT].get()
         current_chat_val = self.window[self.KEY_CHAT_MODEL_SELECT].get()
         self.window[self.KEY_TEST_MODEL_SELECT].update(values=display_values, value=current_test_val if current_test_val in display_values else '')
         self.window[self.KEY_CHAT_MODEL_SELECT].update(values=display_values, value=current_chat_val if current_chat_val in display_values else '')
         # Enable testing if a model is selected
         self.window[self.KEY_TEST_START].update(disabled=not bool(self.window[self.KEY_TEST_MODEL_SELECT].get()))
         self.window[self.KEY_CHAT_START_SESSION].update(disabled=not bool(self.window[self.KEY_CHAT_MODEL_SELECT].get()))


    def _load_test_input(self, values: Dict):
         """Load text from a file into the test input area."""
         file_path = sg.popup_get_file("Select Input File", file_types=(("Text Files", "*.txt"),("All Files", "*.*")))
         if file_path:
             try:
                 with open(file_path, 'r', encoding='utf-8') as f:
                      content = f.read()
                 self.window[self.KEY_TEST_INPUT].update(content)
                 self._update_status(f"Loaded input from {file_path}")
             except Exception as e:
                  self._update_status(f"Error loading file: {e}", error=True)
                  sg.popup_error(f"Error loading file:\n{e}")

    def _test_model_thread(self, values: Dict, status_callback: callable):
         """Thread function to test a model."""
         log = lambda msg: status_callback(msg) if status_callback else print(msg)
         self.window[self.KEY_TEST_OUTPUT_LOG].update('') # Clear log

         display_name = values[self.KEY_TEST_MODEL_SELECT]
         model_name = self.test_model_display_to_id.get(display_name) # Get ID from display name
         test_input = values[self.KEY_TEST_INPUT]
         temp_str = values[self.KEY_TEST_TEMP]
         max_tokens_str = values[self.KEY_TEST_MAX_TOKENS]

         if not model_name: raise ValueError("No model selected.")
         if not test_input: raise ValueError("Input cannot be empty.")

         gen_config = {}
         try:
             gen_config['temperature'] = float(temp_str)
             gen_config['max_output_tokens'] = int(max_tokens_str)
         except ValueError:
             log("Invalid generation parameters. Using defaults.")

         log(f"Sending input to model: {model_name}")
         self.window[self.KEY_TEST_START].update(disabled=True) # Disable button during run

         response = self.model_manager.test_model(model_name, test_input, gen_config)

         # Re-enable button happens via EVENT_THREAD_RESULT handler

         if response is not None:
             log("Response received successfully.")
             return response
         else:
             log("Failed to get response. Check logs/API status.")
             # Return None to indicate failure/no response
             return None


    def _save_test_output(self, values: Dict):
         """Save the content of the test output box to a file."""
         output_text = values[self.KEY_TEST_OUTPUT]
         if not output_text:
              sg.popup_notify("Nothing to save.")
              return

         save_path = sg.popup_get_file("Save Response As", save_as=True, file_types=(("Text Files", "*.txt"),("All Files", "*.*")))
         if save_path:
             try:
                 with open(save_path, 'w', encoding='utf-8') as f:
                      f.write(output_text)
                 self._update_status(f"Response saved to {save_path}", success=True)
                 sg.popup("Response Saved", f"Output saved successfully to:\n{save_path}")
             except Exception as e:
                  self._update_status(f"Error saving response: {e}", error=True)
                  sg.popup_error(f"Error saving response:\n{e}")

    # --- Chat Tab Methods ---
    def _start_chat_session(self, values: Dict):
        if self.active_chat_id:
            confirm = sg.popup_yes_no("A chat session is already active. End it and start a new one?", title="Confirm New Chat")
            if confirm == 'Yes':
                self._end_chat_session()
            else:
                return

        display_name = values[self.KEY_CHAT_MODEL_SELECT]
        model_name = self.test_model_display_to_id.get(display_name) # Use same mapping

        if not model_name:
            sg.popup_error("No valid model selected.", title="Error")
            return

        self._update_chat_status(f"Starting chat with {display_name}...")
        # Run create_chat in main thread as it should be quick
        self.active_chat_id = self.model_manager.create_chat(model_name=model_name)

        if self.active_chat_id:
            self.active_chat_model = display_name
            self._update_chat_status(f"Chatting with {display_name}. Type your message.")
            self.window[self.KEY_CHAT_DISPLAY].update('') # Clear display
            self.window[self.KEY_CHAT_INPUT].update('')
            # Enable/disable controls
            self.window[self.KEY_CHAT_SEND].update(disabled=False)
            self.window[self.KEY_CHAT_INPUT].update(disabled=False)
            self.window[self.KEY_CHAT_END_SESSION].update(disabled=False)
            self.window[self.KEY_CHAT_START_SESSION].update(disabled=True)
            self.window[self.KEY_CHAT_MODEL_SELECT].update(disabled=True)
            self.window[self.KEY_CHAT_REFRESH_MODELS].update(disabled=True)
            self.window[self.KEY_CHAT_INPUT].set_focus()
        else:
            self._update_chat_status("Failed to start chat session.", error=True)
            sg.popup_error("Could not start chat session. Check API key and model validity.", title="Chat Error")


    def _send_chat_message(self, values: Dict):
        if not self.active_chat_id:
            self._update_chat_status("No active chat session.", error=True)
            return

        user_message = values[self.KEY_CHAT_INPUT].strip()
        if not user_message:
            return

        # Display user message
        sg.cprint(f"You: {user_message}\n", key=self.KEY_CHAT_DISPLAY)
        self.window[self.KEY_CHAT_INPUT].update('') # Clear input AFTER sending

        # Disable input while waiting
        self.window[self.KEY_CHAT_SEND].update(disabled=True)
        self.window[self.KEY_CHAT_INPUT].update(disabled=True)
        self._update_chat_status(f"Sending to {self.active_chat_model}...")

        # Start response thread
        thread = threading.Thread(
            target=self._chat_response_thread,
            args=(self.active_chat_id, user_message),
            daemon=True
        )
        thread.start()

    def _chat_response_thread(self, chat_id: str, user_message: str):
        """Thread to handle getting and streaming chat response."""
        if not self.window: return
        try:
            response_stream = self.model_manager.send_message(chat_id, user_message, stream=True)
            if response_stream:
                 self.window.write_event_value(self.EVENT_CHAT_CHUNK, f"Gemini: ") # Start response line
                 full_response_text = ""
                 for chunk in response_stream:
                      self.window.write_event_value(self.EVENT_CHAT_CHUNK, chunk)
                      full_response_text += chunk
                 self.window.write_event_value(self.EVENT_CHAT_CHUNK, "\n") # End response line
                 # Optionally log full response here if needed
                 self.window.write_event_value(self.EVENT_THREAD_STATUS, "Response received.") # Update status bar via event
            else:
                 self.window.write_event_value(self.EVENT_CHAT_CHUNK, "\n[No response or error]\n")
                 self.window.write_event_value(self.EVENT_THREAD_ERROR, "Failed to get chat response.")

        except Exception as e:
             logger.error(f"Chat response error: {e}", exc_info=True)
             self.window.write_event_value(self.EVENT_CHAT_CHUNK, f"\n[Error: {e}]\n")
             self.window.write_event_value(self.EVENT_THREAD_ERROR, f"Chat error: {e}")
        finally:
             # Re-enable input in main thread via event? Or just update here?
             # Let's use status update to re-enable
             self.window.write_event_value('-CHAT_INPUT_ENABLE-', '') # Custom event


    def _end_chat_session(self):
        if self.active_chat_id:
            chat_id_to_delete = self.active_chat_id
            self.active_chat_id = None
            self.active_chat_model = None
            # Run delete in main thread, should be quick
            deleted = self.model_manager.delete_chat(chat_id_to_delete)
            if deleted:
                self._update_chat_status("Chat session ended.")
            else:
                 self._update_chat_status("Chat session not found or already ended.", error=True)
        else:
             self._update_chat_status("No active chat session to end.")

        # Reset UI elements
        self.window[self.KEY_CHAT_SEND].update(disabled=True)
        self.window[self.KEY_CHAT_INPUT].update(disabled=True, value='')
        self.window[self.KEY_CHAT_END_SESSION].update(disabled=True)
        self.window[self.KEY_CHAT_START_SESSION].update(disabled=not bool(self.window[self.KEY_CHAT_MODEL_SELECT].get())) # Re-enable if model selected
        self.window[self.KEY_CHAT_MODEL_SELECT].update(disabled=False)
        self.window[self.KEY_CHAT_REFRESH_MODELS].update(disabled=False)

    def _update_chat_status(self, message: str, error: bool = False):
        color = 'red' if error else 'black'
        self.window[self.KEY_CHAT_STATUS].update(message, text_color=color)
        # Also update main status bar
        self._update_status(f"Chat: {message}", error=error)



# --- Main Execution ---

def check_requirements() -> List[str]:
    """Check for missing required packages."""
    missing = []
    if not HAS_GENAI: missing.append("google-genai")
    if not HAS_DOTENV: missing.append("python-dotenv")
    if not HAS_TQDM: missing.append("tqdm")
    # GUI is optional, but check if --gui flag is used later
    # if not HAS_GUI: missing.append("PySimpleGUI")
    return missing

def install_packages(packages: List[str]) -> bool:
    """Attempt to install missing packages using pip."""
    if not packages: return True
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        logger.info(f"Attempting to install: {' '.join(packages)}")
        subprocess.check_call(cmd)
        logger.info("Installation successful.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install packages using pip: {e}")
        logger.error("Please install them manually:")
        logger.error(f"  pip install {' '.join(packages)}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during installation: {e}")
        return False

def main() -> None:
    """Main function to parse arguments and run the appropriate interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced Gemini Tuner: Process data and tune Gemini models via CLI, GUI, or Interactive mode.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Mode Selection ---
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--gui", action="store_true", help="Run in graphical user interface mode (requires PySimpleGUI).")
    mode_group.add_argument("--interactive", action="store_true", help="Run in interactive command-line mode.")

    # --- Common Options ---
    parser.add_argument("--config", type=str, help="Path to a specific .env configuration file.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose DEBUG logging.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress INFO logging, show only warnings and errors.")
    parser.add_argument("--version", action="version", version="Gemini Tuner v1.1.0") # Update version as needed

    # --- Subcommands for Non-Interactive CLI ---
    subparsers = parser.add_subparsers(dest="command", help="Run a specific command directly (non-interactive CLI mode). See '<command> --help'.")

    # --- Process Command ---
    process_parser = subparsers.add_parser("process", help="Process input text file(s) into JSONL training data.")
    process_parser.add_argument("input_files", type=str, nargs='+', help="Path(s) to input text file(s).")
    process_parser.add_argument("--output", "-o", type=str, default=f"training_data_{datetime.now().strftime('%Y%m%d')}.jsonl", help="Output JSONL file path.")
    process_parser.add_argument("--delimiter", type=str, help="Example delimiter (overrides config).")
    process_parser.add_argument("--input-prefix", type=str, help="Input prefix (overrides config).")
    process_parser.add_argument("--output-prefix", type=str, help="Output prefix (overrides config).")
    process_parser.add_argument("--preview", action="store_true", help="Preview first few processed examples to console.")

    # --- Train Command ---
    train_parser = subparsers.add_parser("train", help="Start a model tuning job using a JSONL data file.")
    train_parser.add_argument("--data", "-d", type=str, required=True, help="Path to the training data JSONL file.")
    train_parser.add_argument("--base-model", "-m", type=str, help="Base model name to tune (e.g., models/gemini-1.0-pro-001, overrides config).")
    train_parser.add_argument("--name", "-n", type=str, help="Display name for the tuned model (defaults to auto-generated).")
    train_parser.add_argument("--epochs", "-e", type=int, help="Number of training epochs (overrides config).")
    train_parser.add_argument("--batch-size", "-b", type=int, help="Batch size for training (overrides config).")
    train_parser.add_argument("--learning-rate", "-lr", type=float, dest="learning_rate_multiplier", help="Learning rate multiplier (overrides config).")

    # --- Test Command ---
    test_parser = subparsers.add_parser("test", help="Test a model (base or tuned) with a prompt.")
    test_parser.add_argument("model_name", type=str, help="Full name/ID of the model to test (e.g., models/gemini..., tunedModels/abc-123).")
    test_parser.add_argument("--input", "-i", type=str, required=True, help="Input prompt text OR path to a file containing the prompt.")
    test_parser.add_argument("--output", "-o", type=str, help="Optional file path to save the model's response.")
    test_parser.add_argument("--temp", type=float, default=0.7, help="Generation temperature.")
    test_parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum output tokens.")

    # --- Models Command ---
    models_cmd = subparsers.add_parser("models", help="Manage tuned models.")
    models_group = models_cmd.add_mutually_exclusive_group(required=True)
    models_group.add_argument("--list", action="store_true", help="List all tuned models for your project.")
    models_group.add_argument("--delete", type=str, metavar="MODEL_NAME", help="Delete a tuned model by its full name (ID, e.g., tunedModels/abc-123).")
    models_group.add_argument("--status", type=str, metavar="MODEL_NAME", help="Check the status of a tuned model by its full name (ID).")

    # --- Setup Command ---
    setup_cmd = subparsers.add_parser("setup", help="Utility commands for setup.")
    setup_group = setup_cmd.add_mutually_exclusive_group(required=True)
    setup_group.add_argument("--create-sample-config", action="store_true", help="Create a sample .env.example configuration file.")
    setup_group.add_argument("--check-requirements", action="store_true", help="Check if required Python packages are installed.")
    setup_group.add_argument("--validate-api-key", action="store_true", help="Attempt to validate the GOOGLE_API_KEY found in config/env.")


    # --- Parse Arguments ---
    args = parser.parse_args()

    # --- Configure Logging Level ---
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # --- Handle Requirements Check ---
    missing = check_requirements()
    if missing:
        logger.warning(f"Missing required packages: {', '.join(missing)}")
        # If GUI is requested but missing, attempt install or exit/warn
        if args.gui and not HAS_GUI:
             logger.info("PySimpleGUI is required for GUI mode.")
             if install_packages(["PySimpleGUI"]):
                  try:
                       import PySimpleGUI as sg
                       globals()['HAS_GUI'] = True
                       globals()['sg'] = sg
                       logger.info("PySimpleGUI installed successfully.")
                  except ImportError:
                       logger.error("Failed to import PySimpleGUI even after install attempt. GUI mode unavailable.")
                       sys.exit(1)
             else:
                  logger.error("Failed to install PySimpleGUI. Cannot run in GUI mode.")
                  sys.exit(1)
        # For other missing packages, attempt install if not quiet mode
        elif not args.quiet:
             if install_packages(missing):
                  logger.info("Attempting to reload modules after installation...")
                  # Trying to reload dynamically is complex and often fails reliably.
                  # Best practice is usually to ask the user to re-run the script.
                  logger.warning("Packages installed. Please re-run the script for changes to take effect.")
                  sys.exit(0) # Exit after install attempt
             else:
                  # Install failed, continue with limited functionality if possible
                  logger.error("Continuing with potentially limited functionality due to missing packages.")


    # --- Initialize Core Components ---
    config = Config(args.config)
    processor = DataProcessor(config)
    model_manager = ModelManager(config)

    # --- Execute Based on Mode/Command ---

    # GUI Mode
    if args.gui:
        if not HAS_GUI: # Double check after potential install attempt
             logger.error("PySimpleGUI is still not available. Cannot run GUI.")
             sys.exit(1)
        logger.info("Launching Graphical User Interface...")
        gui = GUI(config, processor, model_manager)
        gui.run()

    # Interactive CLI Mode
    elif args.interactive:
        logger.info("Starting Interactive CLI Mode...")
        interactive_cli = InteractiveCLI(config, processor, model_manager)
        interactive_cli.run()

    # Direct Command Mode
    elif args.command:
        logger.debug(f"Executing command: {args.command}")

        if args.command == "process":
            try:
                processed_data, valid_count, invalid_count = processor.process_multiple_files(
                    args.input_files,
                    delimiter=args.delimiter, # Pass overrides or None
                    input_prefix=args.input_prefix,
                    output_prefix=args.output_prefix
                )
                if valid_count > 0:
                    if args.preview:
                        processor.preview_training_data(processed_data)
                    processor.save_training_data(processed_data, args.output)
                elif invalid_count > 0:
                     logger.warning("Processing finished, but no valid examples found.")
                else:
                     logger.error("Processing failed. Check file paths and format.")

            except Exception as e:
                logger.error(f"Error during file processing: {e}", exc_info=args.verbose)
                sys.exit(1)

        elif args.command == "train":
            if not model_manager.is_client_ready():
                 logger.error("Cannot execute 'train' command: API key not configured or invalid.")
                 sys.exit(1)
            try:
                training_data = processor.load_training_data(args.data)
                if not training_data:
                    logger.error(f"No valid training data found in {args.data}")
                    sys.exit(1)

                submitted_ref = model_manager.train_model(
                    training_data=training_data,
                    base_model=args.base_model, # Pass override or None
                    display_name=args.name,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate_multiplier=args.learning_rate_multiplier
                )
                if not submitted_ref:
                     sys.exit(1) # Error already logged by model_manager

            except FileNotFoundError as e:
                 logger.error(f"Training data file not found: {e}")
                 sys.exit(1)
            except Exception as e:
                logger.error(f"Error during training submission: {e}", exc_info=args.verbose)
                sys.exit(1)

        elif args.command == "test":
            if not model_manager.is_client_ready():
                 logger.error("Cannot execute 'test' command: API key not configured or invalid.")
                 sys.exit(1)
            try:
                test_input = args.input
                input_path = Path(args.input)
                if input_path.is_file():
                     logger.info(f"Reading test input from file: {input_path}")
                     with open(input_path, 'r', encoding='utf-8') as f:
                          test_input = f.read()

                gen_config = {"temperature": args.temp, "max_output_tokens": args.max_tokens}

                response = model_manager.test_model(args.model_name, test_input, gen_config)

                if response is not None:
                     print("\n--- Model Response ---")
                     print(response)
                     print("--- End Response ---")
                     if args.output:
                          try:
                              with open(args.output, 'w', encoding='utf-8') as f:
                                   f.write(response)
                              logger.info(f"Response saved to {args.output}")
                          except IOError as e:
                              logger.error(f"Error saving response to {args.output}: {e}")
                else:
                     logger.error("Model did not return a response (check logs for potential blocking or errors).")
                     sys.exit(1)

            except Exception as e:
                logger.error(f"Error during model testing: {e}", exc_info=args.verbose)
                sys.exit(1)

        elif args.command == "models":
            if not model_manager.is_client_ready():
                 logger.error("Cannot execute 'models' command: API key not configured or invalid.")
                 sys.exit(1)
            try:
                if args.list:
                    models = model_manager.list_models(only_tuned=True)
                    if models:
                        print("\n--- Tuned Models ---")
                        for i, m in enumerate(models):
                            ct = m.get('create_time')
                            ct_str = ct.strftime('%Y-%m-%d %H:%M') if hasattr(ct, 'strftime') else str(ct)
                            print(f"{i+1}. {m.get('display_name')} (State: {m.get('state', '').upper()})")
                            print(f"   ID: {m.get('name')}")
                            print(f"   Base: {m.get('base_model')}")
                            print(f"   Created: {ct_str}")
                        print("--------------------")
                    else:
                        print("No tuned models found.")
                elif args.delete:
                    if not model_manager.delete_model(args.delete):
                         sys.exit(1) # Error logged by manager
                elif args.status:
                     status = model_manager.get_tuning_job_status(args.status)
                     if status:
                          ct = status.get('create_time')
                          ut = status.get('update_time')
                          ct_str = ct.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ct, 'strftime') else str(ct)
                          ut_str = ut.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ut, 'strftime') else str(ut)
                          print("\n--- Model Status ---")
                          print(f"Display Name: {status.get('display_name')}")
                          print(f"ID:           {status.get('name')}")
                          print(f"State:        {status.get('state', '').upper()}")
                          print(f"Created:      {ct_str}")
                          print(f"Updated:      {ut_str}")
                          print("--------------------")
                     else:
                          sys.exit(1) # Error logged by manager

            except Exception as e:
                logger.error(f"Error during model management: {e}", exc_info=args.verbose)
                sys.exit(1)

        elif args.command == "setup":
             try:
                  if args.create_sample_config:
                       Config.create_sample_config()
                  elif args.check_requirements:
                       missing = check_requirements()
                       if missing:
                            print(f"Missing required packages: {', '.join(missing)}")
                            print(f"Install using: pip install {' '.join(missing)}")
                       else:
                            print("All required packages seem to be installed.")
                       # Check optional GUI package
                       if not HAS_GUI:
                            print("Optional package for GUI mode missing: PySimpleGUI")
                            print("Install using: pip install PySimpleGUI")
                  elif args.validate_api_key:
                       api_key = config.get('api_key')
                       if not api_key:
                            print("No GOOGLE_API_KEY found in config or environment.")
                       else:
                            print("Validating API key...")
                            if model_manager.validate_api_key(api_key):
                                 print("API key appears valid.")
                            else:
                                 print("API key validation failed (invalid key or permissions?).")
             except Exception as e:
                  logger.error(f"Error during setup command: {e}", exc_info=args.verbose)
                  sys.exit(1)


    # No mode or command specified
    else:
        parser.print_help()
        print("\nChoose a mode (--gui, --interactive) or specify a command.")

if __name__ == "__main__":
    main()
