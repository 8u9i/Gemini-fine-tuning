#!/usr/bin/env python3
"""
Gemini Model Tuning with Flexible Text Storage

This script implements a complete training pipeline for fine-tuning a Gemini model
using text data that can be stored either in a PostgreSQL database or locally in files.

Environment variables required:
- GOOGLE_API_KEY: Your Google API key for Gemini
- STORAGE_MODE: Where to store training data ('database' or 'local')
- DATABASE_URL: PostgreSQL connection string (required if STORAGE_MODE=database)
- LOCAL_DATA_DIR: Directory for local text storage (required if STORAGE_MODE=local)
- TUNED_MODEL_NAME: Name for your tuned model (optional)

Usage:
  python gemini_tuning_flexible_storage.py [--setup-storage] [--import-sample-data] [--tune-model] [--test-model]

Options:
  --setup-storage       Create the necessary database tables or local directories
  --import-sample-data  Import sample text data into storage
  --tune-model          Run the model tuning process
  --test-model          Test the tuned model with sample prompts
  
If no options are provided, all steps will be executed in sequence.
"""

import os
import sys
import json
import time
import logging
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from urllib.parse import urlparse

# Import database libraries conditionally
try:
    import psycopg2
    from psycopg2.extras import DictCursor
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gemini_tuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gemini_tuning")

# Sample text data for demonstration
SAMPLE_TEXT_DATA = [
    """The solar system consists of the Sun and everything that orbits around it, including planets, dwarf planets, moons, asteroids, comets, and meteoroids. The Sun is at the center of our solar system and contains about 99.8% of the solar system's mass. Eight planets orbit the Sun: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. These planets, along with dwarf planets like Pluto, orbit the Sun in elliptical paths.""",
    
    """Machine learning is a subset of artificial intelligence that focuses on developing systems that learn from data. Instead of explicitly programming rules, machine learning algorithms identify patterns in data and make predictions or decisions. Common types include supervised learning (using labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error with rewards).""",
    
    """The Great Barrier Reef is the world's largest coral reef system, composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometers. Located in the Coral Sea off the coast of Queensland, Australia, it can be seen from outer space and is the world's biggest single structure made by living organisms. The reef is home to thousands of species including fish, coral, mollusks, sea turtles, and dolphins.""",
    
    """JavaScript is a high-level, interpreted programming language that conforms to the ECMAScript specification. It has curly-bracket syntax, dynamic typing, prototype-based object-orientation, and first-class functions. Alongside HTML and CSS, JavaScript is one of the core technologies of the World Wide Web, enabling interactive web pages and being an essential part of web applications.""",
    
    """Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil, and gas, which produces heat-trapping gases. The main effects include rising temperatures, shifting precipitation patterns, and more frequent extreme weather events.""",
    
    """The human brain is the central organ of the human nervous system and is located in the head, protected by the skull. It has the same general structure as the brains of other mammals, but is larger in relation to body size than any other brains. The human brain controls nearly all of the body's activities, processing, integrating, and coordinating information from the sensory organs, and making decisions as to the instructions sent to the rest of the body.""",
    
    """Photosynthesis is the process used by plants, algae, and certain bacteria to harness energy from sunlight and turn it into chemical energy. During photosynthesis, these organisms capture light energy and use it to power chemical reactions that convert carbon dioxide and water into oxygen and energy-rich carbohydrates like sugars and starches. This process is essential for maintaining the oxygen level in the atmosphere.""",
    
    """The Renaissance was a period in European history marking the transition from the Middle Ages to modernity and covering the 15th and 16th centuries. It began in Italy and spread to the rest of Europe. The Renaissance is characterized by an emphasis on the individual, a revival of classical learning, and a more scientific approach to examining the natural world. It saw developments in art, architecture, politics, science, and literature.""",
    
    """Blockchain is a distributed database or ledger that is shared among the nodes of a computer network. As a database, a blockchain stores information electronically in digital format. Blockchains are best known for their crucial role in cryptocurrency systems, such as Bitcoin, for maintaining a secure and decentralized record of transactions. The innovation with a blockchain is that it guarantees the fidelity and security of a record of data and generates trust without the need for a trusted third party.""",
    
    """The immune system is a complex network of cells, tissues, and organs that work together to defend the body against harmful invaders. These invaders can include bacteria, viruses, parasites, and fungi. The immune system has two main parts: the innate immune system, which you are born with, and the adaptive immune system, which develops as you are exposed to microbes. Together, these two systems create a multilayered defense against pathogens.""",
    
    """Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations. The devices that perform quantum computations are known as quantum computers. They are believed to be able to solve certain computational problems, such as integer factorization, substantially faster than classical computers.""",
    
    """The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups from the Eurasian Steppe. Several walls were built from as early as the 7th century BC, with selective stretches later joined together by Qin Shi Huang (220–206 BC), the first emperor of China. Later on, many successive dynasties built and maintained multiple stretches of border walls. The most well-known sections of the wall were built by the Ming dynasty (1368–1644).""",
    
    """Democracy is a form of government in which the people have the authority to deliberate and decide legislation, or to choose governing officials to do so. Who is considered part of "the people" and how authority is shared among them has changed over time and at different rates in different countries, but over time more and more of a democratic country's inhabitants have generally been included. Cornerstones of democracy include freedom of assembly, association, property rights, freedom of religion and speech, inclusiveness and equality, citizenship, consent of the governed, voting rights, freedom from unwarranted governmental deprivation of the right to life and liberty, and minority rights.""",
    
    """The theory of relativity, developed by Albert Einstein, describes the physics of motion in the absence of gravitational fields (special relativity) and in the presence of gravitational fields (general relativity). Special relativity, introduced in 1905, deals with the relationship between space and time and provides a theoretical framework for understanding electromagnetic phenomena. General relativity, published in 1915, explains gravity as a geometric property of space and time, or spacetime, and has important implications for cosmology and our understanding of the universe.""",
    
    """Artificial neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains. An ANN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal to other neurons. An artificial neuron receives signals then processes them and can signal neurons connected to it. The "signal" at a connection is a real number, and the output of each neuron is computed by some non-linear function of the sum of its inputs."""
]

class StorageManager:
    """Base class for storage managers."""
    
    def setup_storage(self):
        """Set up the storage system."""
        raise NotImplementedError("Subclasses must implement setup_storage")
    
    def import_sample_data(self):
        """Import sample text data into storage."""
        raise NotImplementedError("Subclasses must implement import_sample_data")
    
    def get_training_examples(self, limit=None, offset=0, category=None):
        """Retrieve text training examples from storage."""
        raise NotImplementedError("Subclasses must implement get_training_examples")
    
    def get_example_count(self, category=None):
        """Get the total count of text training examples in storage."""
        raise NotImplementedError("Subclasses must implement get_example_count")
    
    def add_training_example(self, text, category=None):
        """Add a new training example to storage."""
        raise NotImplementedError("Subclasses must implement add_training_example")
    
    def record_tuning_job(self, model_name, base_model, examples_count, config):
        """Record a new tuning job."""
        raise NotImplementedError("Subclasses must implement record_tuning_job")
    
    def update_tuning_job_status(self, job_id, status, completed_at=None):
        """Update the status of a tuning job."""
        raise NotImplementedError("Subclasses must implement update_tuning_job_status")

class PostgresManager(StorageManager):
    """Manages PostgreSQL database connections and operations for text training data."""
    
    def __init__(self):
        """Initialize the database manager using DATABASE_URL environment variable."""
        if not HAS_POSTGRES:
            raise ImportError("psycopg2 is required for PostgreSQL storage. Install it with 'pip install psycopg2-binary'")
            
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is not set")
        
        # Parse the DATABASE_URL
        parsed_url = urlparse(database_url)
        
        # Extract connection parameters from the URL
        self.db_params = {
            'dbname': parsed_url.path[1:],  # Remove leading slash
            'user': parsed_url.username,
            'password': parsed_url.password,
            'host': parsed_url.hostname,
            'port': parsed_url.port or 5432
        }
        
        logger.info(f"Initialized PostgreSQL manager with host={self.db_params['host']}, "
                   f"port={self.db_params['port']}, dbname={self.db_params['dbname']}")
    
    def get_connection(self):
        """Get a connection to the PostgreSQL database."""
        try:
            conn = psycopg2.connect(**self.db_params)
            return conn
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def setup_storage(self):
        """Create the necessary tables for text training data."""
        try:
            conn = self.get_connection()
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Create text training data table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS text_training_data (
                id SERIAL PRIMARY KEY,
                text_content TEXT NOT NULL,
                category VARCHAR(100) DEFAULT 'general',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_verified BOOLEAN DEFAULT TRUE
            )
            ''')
            
            # Create tuning jobs table to track model tuning history
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS tuning_jobs (
                id SERIAL PRIMARY KEY,
                model_name TEXT NOT NULL,
                base_model TEXT NOT NULL,
                examples_count INTEGER NOT NULL,
                config JSONB NOT NULL,
                status VARCHAR(50) DEFAULT 'created',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
            ''')
            
            cursor.close()
            conn.close()
            logger.info("Database tables created successfully")
            return True
        except psycopg2.Error as e:
            logger.error(f"Database setup error: {e}")
            return False
    
    def import_sample_data(self):
        """Import sample text data into the database."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Check if data already exists
            cursor.execute("SELECT COUNT(*) FROM text_training_data")
            count = cursor.fetchone()[0]
            
            if count > 0:
                logger.info(f"Database already contains {count} text examples, skipping import")
                cursor.close()
                conn.close()
                return count
            
            # Import sample data
            for text in SAMPLE_TEXT_DATA:
                cursor.execute(
                    "INSERT INTO text_training_data (text_content) VALUES (%s)",
                    (text,)
                )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Imported {len(SAMPLE_TEXT_DATA)} sample text examples into database")
            return len(SAMPLE_TEXT_DATA)
        except psycopg2.Error as e:
            logger.error(f"Data import error: {e}")
            return 0
    
    def get_training_examples(self, limit=None, offset=0, category=None):
        """Retrieve text training examples from the database with optional filtering."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = "SELECT text_content FROM text_training_data WHERE 1=1"
            params = []
            
            if category:
                query += " AND category = %s"
                params.append(category)
            
            query += " ORDER BY id"
            
            if limit is not None:
                query += " LIMIT %s OFFSET %s"
                params.extend([limit, offset])
            
            cursor.execute(query, params)
            examples = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            # Convert from list of tuples to list of strings
            examples = [example[0] for example in examples]
            
            logger.info(f"Retrieved {len(examples)} text examples from database")
            return examples
        except psycopg2.Error as e:
            logger.error(f"Error retrieving training examples: {e}")
            return []
    
    def get_example_count(self, category=None):
        """Get the total count of text training examples in the database with optional filtering."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = "SELECT COUNT(*) FROM text_training_data WHERE 1=1"
            params = []
            
            if category:
                query += " AND category = %s"
                params.append(category)
            
            cursor.execute(query, params)
            count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return count
        except psycopg2.Error as e:
            logger.error(f"Error counting examples: {e}")
            return 0
    
    def add_training_example(self, text, category=None):
        """Add a new training example to the database."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if category:
                cursor.execute(
                    "INSERT INTO text_training_data (text_content, category) VALUES (%s, %s) RETURNING id",
                    (text, category)
                )
            else:
                cursor.execute(
                    "INSERT INTO text_training_data (text_content) VALUES (%s) RETURNING id",
                    (text,)
                )
            
            example_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Added new training example with ID {example_id}")
            return example_id
        except psycopg2.Error as e:
            logger.error(f"Error adding training example: {e}")
            return None
    
    def record_tuning_job(self, model_name, base_model, examples_count, config):
        """Record a new tuning job in the database."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO tuning_jobs (model_name, base_model, examples_count, config, status) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (model_name, base_model, examples_count, json.dumps(config), 'started')
            )
            
            job_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Recorded tuning job with ID {job_id} for model {model_name}")
            return job_id
        except psycopg2.Error as e:
            logger.error(f"Error recording tuning job: {e}")
            return None
    
    def update_tuning_job_status(self, job_id, status, completed_at=None):
        """Update the status of a tuning job."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if completed_at is None and status in ('completed', 'failed'):
                completed_at = datetime.now()
            
            if completed_at:
                cursor.execute(
                    "UPDATE tuning_jobs SET status = %s, completed_at = %s WHERE id = %s",
                    (status, completed_at, job_id)
                )
            else:
                cursor.execute(
                    "UPDATE tuning_jobs SET status = %s WHERE id = %s",
                    (status, job_id)
                )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Updated tuning job {job_id} status to {status}")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error updating tuning job status: {e}")
            return False

class LocalStorageManager(StorageManager):
    """Manages local file storage for text training data."""
    
    def __init__(self):
        """Initialize the local storage manager."""
        self.data_dir = os.environ.get('LOCAL_DATA_DIR')
        if not self.data_dir:
            raise ValueError("LOCAL_DATA_DIR environment variable is not set")
        
        self.texts_dir = os.path.join(self.data_dir, 'texts')
        self.jobs_dir = os.path.join(self.data_dir, 'jobs')
        
        logger.info(f"Initialized local storage manager with data directory: {self.data_dir}")
    
    def setup_storage(self):
        """Create the necessary directories for local storage."""
        try:
            # Create main directories
            os.makedirs(self.texts_dir, exist_ok=True)
            os.makedirs(self.jobs_dir, exist_ok=True)
            
            # Create category directories
            os.makedirs(os.path.join(self.texts_dir, 'general'), exist_ok=True)
            
            # Create a file to track the next ID
            id_tracker_path = os.path.join(self.data_dir, 'id_tracker.json')
            if not os.path.exists(id_tracker_path):
                with open(id_tracker_path, 'w') as f:
                    json.dump({'next_text_id': 1, 'next_job_id': 1}, f)
            
            logger.info("Local storage directories created successfully")
            return True
        except Exception as e:
            logger.error(f"Local storage setup error: {e}")
            return False
    
    def _get_next_id(self, id_type):
        """Get the next available ID for the specified type."""
        id_tracker_path = os.path.join(self.data_dir, 'id_tracker.json')
        
        with open(id_tracker_path, 'r') as f:
            id_tracker = json.load(f)
        
        next_id = id_tracker[f'next_{id_type}_id']
        id_tracker[f'next_{id_type}_id'] = next_id + 1
        
        with open(id_tracker_path, 'w') as f:
            json.dump(id_tracker, f)
        
        return next_id
    
    def import_sample_data(self):
        """Import sample text data into local storage."""
        try:
            # Check if data already exists
            existing_files = list(Path(os.path.join(self.texts_dir, 'general')).glob('*.txt'))
            if existing_files:
                logger.info(f"Local storage already contains {len(existing_files)} text examples, skipping import")
                return len(existing_files)
            
            # Import sample data
            count = 0
            for text in SAMPLE_TEXT_DATA:
                self.add_training_example(text)
                count += 1
            
            logger.info(f"Imported {count} sample text examples into local storage")
            return count
        except Exception as e:
            logger.error(f"Data import error: {e}")
            return 0
    
    def get_training_examples(self, limit=None, offset=0, category=None):
        """Retrieve text training examples from local storage with optional filtering."""
        try:
            category_dir = os.path.join(self.texts_dir, category or 'general')
            
            # Get all text files in the category directory
            text_files = sorted(Path(category_dir).glob('*.txt'), key=lambda p: int(p.stem.split('_')[0]))
            
            # Apply offset and limit
            if offset > 0:
                text_files = text_files[offset:]
            if limit is not None:
                text_files = text_files[:limit]
            
            # Read the content of each file
            examples = []
            for file_path in text_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    examples.append(f.read())
            
            logger.info(f"Retrieved {len(examples)} text examples from local storage")
            return examples
        except Exception as e:
            logger.error(f"Error retrieving training examples: {e}")
            return []
    
    def get_example_count(self, category=None):
        """Get the total count of text training examples in local storage with optional filtering."""
        try:
            category_dir = os.path.join(self.texts_dir, category or 'general')
            
            # Count all text files in the category directory
            count = len(list(Path(category_dir).glob('*.txt')))
            
            return count
        except Exception as e:
            logger.error(f"Error counting examples: {e}")
            return 0
    
    def add_training_example(self, text, category=None):
        """Add a new training example to local storage."""
        try:
            category = category or 'general'
            category_dir = os.path.join(self.texts_dir, category)
            
            # Create category directory if it doesn't exist
            os.makedirs(category_dir, exist_ok=True)
            
            # Get the next ID
            example_id = self._get_next_id('text')
            
            # Create a metadata object
            metadata = {
                'id': example_id,
                'category': category,
                'created_at': datetime.now().isoformat(),
                'is_verified': True
            }
            
            # Save the text file
            file_path = os.path.join(category_dir, f"{example_id:06d}_{category}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Save the metadata file
            metadata_path = os.path.join(category_dir, f"{example_id:06d}_{category}.meta.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Added new training example with ID {example_id}")
            return example_id
        except Exception as e:
            logger.error(f"Error adding training example: {e}")
            return None
    
    def record_tuning_job(self, model_name, base_model, examples_count, config):
        """Record a new tuning job in local storage."""
        try:
            # Get the next job ID
            job_id = self._get_next_id('job')
            
            # Create a job record
            job_record = {
                'id': job_id,
                'model_name': model_name,
                'base_model': base_model,
                'examples_count': examples_count,
                'config': config,
                'status': 'started',
                'created_at': datetime.now().isoformat(),
                'completed_at': None
            }
            
            # Save the job record
            job_path = os.path.join(self.jobs_dir, f"job_{job_id:06d}.json")
            with open(job_path, 'w', encoding='utf-8') as f:
                json.dump(job_record, f, indent=2)
            
            logger.info(f"Recorded tuning job with ID {job_id} for model {model_name}")
            return job_id
        except Exception as e:
            logger.error(f"Error recording tuning job: {e}")
            return None
    
    def update_tuning_job_status(self, job_id, status, completed_at=None):
        """Update the status of a tuning job."""
        try:
            # Load the job record
            job_path = os.path.join(self.jobs_dir, f"job_{job_id:06d}.json")
            with open(job_path, 'r', encoding='utf-8') as f:
                job_record = json.load(f)
            
            # Update the status
            job_record['status'] = status
            
            # Update the completion time if provided or if the status is 'completed' or 'failed'
            if completed_at is None and status in ('completed', 'failed'):
                completed_at = datetime.now().isoformat()
            
            if completed_at:
                job_record['completed_at'] = completed_at
            
            # Save the updated job record
            with open(job_path, 'w', encoding='utf-8') as f:
                json.dump(job_record, f, indent=2)
            
            logger.info(f"Updated tuning job {job_id} status to {status}")
            return True
        except Exception as e:
            logger.error(f"Error updating tuning job status: {e}")
            return False

class GeminiTuningManager:
    """Manages the Gemini model tuning process for text-only training."""
    
    def __init__(self, storage_manager):
        """Initialize the tuning manager.
        
        Args:
            storage_manager: StorageManager instance
        """
        self.storage_manager = storage_manager
        self.config = self._get_config()
        self.client = None
        self._initialize_client()
    
    def _get_config(self):
        """Get configuration values from environment variables or defaults."""
        return {
            "base_model": "models/gemini-1.5-flash-001-tuning",
            "batch_size": int(os.environ.get('TUNING_BATCH_SIZE', '4')),
            "epoch_count": int(os.environ.get('TUNING_EPOCH_COUNT', '5')),
            "learning_rate": float(os.environ.get('TUNING_LEARNING_RATE', '0.001')),
            "tuned_model_display_name": os.environ.get('TUNED_MODEL_NAME', 
                                                      f"text_tuned_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            "processing_batch_size": int(os.environ.get('PROCESSING_BATCH_SIZE', '100')),
            # For text-only training, we'll use the same text as both input and output
            "input_prefix": os.environ.get('INPUT_PREFIX', 'Continue the following text: '),
            "output_length_ratio": float(os.environ.get('OUTPUT_LENGTH_RATIO', '0.5'))
        }
    
    def _initialize_client(self):
        """Initialize the Gemini API client."""
        try:
            api_key = os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set")
            
            genai.configure(api_key=api_key)
            self.client = genai.Client()
            logger.info("Successfully initialized Gemini client")
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")
            raise
    
    def _prepare_text_example(self, text):
        """Prepare a text example for training by splitting it into input and output parts.
        
        For text-only training, we'll use a portion of the text as input and the rest as output.
        
        Args:
            text: The full text content
            
        Returns:
            Tuple of (input_text, output_text)
        """
        # Calculate the split point based on the output_length_ratio
        # For example, if ratio is 0.5, we'll use the first half as input and second half as output
        split_point = int(len(text) * (1 - self.config["output_length_ratio"]))
        
        # Ensure we have at least some content for both input and output
        split_point = max(min(split_point, len(text) - 50), 50)
        
        # Split the text
        input_text = self.config["input_prefix"] + text[:split_point]
        output_text = text[split_point:]
        
        return input_text, output_text
    
    def create_tuning_dataset(self, texts):
        """Create a TuningDataset from text examples.
        
        Args:
            texts: List of text strings
            
        Returns:
            TuningDataset object
        """
        try:
            # Prepare examples by splitting each text into input and output parts
            examples = [self._prepare_text_example(text) for text in texts]
            
            dataset = types.TuningDataset(
                examples=[
                    types.TuningExample(
                        text_input=input_text,
                        output=output_text,
                    )
                    for input_text, output_text in examples
                ],
            )
            logger.info(f"Created TuningDataset with {len(examples)} text examples")
            return dataset
        except Exception as e:
            logger.error(f"Error creating TuningDataset: {e}")
            raise
    
    def tune_model(self):
        """Tune the model using text examples from storage.
        
        Returns:
            Tuning job object
        """
        # Get total count of examples
        total_examples = self.storage_manager.get_example_count()
        logger.info(f"Starting model tuning with {total_examples} total text examples")
        
        if total_examples == 0:
            raise ValueError("No training examples found in storage")
        
        # For small datasets, process all at once
        if total_examples <= self.config["processing_batch_size"]:
            examples = self.storage_manager.get_training_examples()
            training_dataset = self.create_tuning_dataset(examples)
        else:
            # For larger datasets, process in batches
            logger.info(f"Processing large dataset in batches of {self.config['processing_batch_size']}")
            all_examples = []
            offset = 0
            
            while offset < total_examples:
                batch = self.storage_manager.get_training_examples(
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
            
            # Record the tuning job in storage
            job_id = self.storage_manager.record_tuning_job(
                tuning_job.tuned_model.model,
                self.config["base_model"],
                total_examples,
                self.config
            )
            
            # Save the model name to a file for easy reference
            with open('tuned_model_name.txt', 'w') as f:
                f.write(tuning_job.tuned_model.model)
            
            return tuning_job
        except Exception as e:
            logger.error(f"Error starting tuning job: {e}")
            raise
    
    def test_model(self, model_name=None, test_prompts=None):
        """Test the tuned model with sample prompts.
        
        Args:
            model_name: Name of the tuned model (or None to use the latest)
            test_prompts: List of prompts to test (or None to use defaults)
            
        Returns:
            Dictionary mapping prompts to model responses
        """
        # If no model name provided, try to load from file
        if not model_name:
            try:
                with open('tuned_model_name.txt', 'r') as f:
                    model_name = f.read().strip()
            except FileNotFoundError:
                # If file not found, list models and use the first tuned model
                models = list(self.client.models.list())
                tuned_models = [m for m in models if "tuned" in m.name.lower()]
                if tuned_models:
                    model_name = tuned_models[0].name
                else:
                    raise ValueError("No tuned model found. Please run tuning first or specify a model name.")
        
        # Default test prompts if none provided
        if not test_prompts:
            test_prompts = [
                "Continue the following text: The solar system consists of the Sun and everything that orbits around it,",
                "Continue the following text: Machine learning is a subset of artificial intelligence that",
                "Continue the following text: Blockchain is a distributed database or ledger that",
                "Continue the following text: The theory of relativity, developed by Albert Einstein,",
                "Continue the following text: Artificial neural networks are computing systems"
            ]
        
        results = {}
        
        for prompt in test_prompts:
            try:
                logger.info(f"Testing model with prompt: '{prompt}'")
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                results[prompt] = response.text
                logger.info(f"Model response: '{response.text}'")
            except Exception as e:
                logger.error(f"Error testing model with prompt '{prompt}': {e}")
                results[prompt] = f"ERROR: {str(e)}"
        
        # Save test results to a file
        with open('text_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def create_storage_manager():
    """Create the appropriate storage manager based on the STORAGE_MODE environment variable."""
    storage_mode = os.environ.get('STORAGE_MODE', 'database').lower()
    
    if storage_mode == 'database':
        if not HAS_POSTGRES:
            logger.error("PostgreSQL support is not available. Install psycopg2 with 'pip install psycopg2-binary'")
            raise ImportError("psycopg2 is required for database storage")
        
        if not os.environ.get('DATABASE_URL'):
            logger.error("DATABASE_URL environment variable is required for database storage")
            raise ValueError("DATABASE_URL environment variable is not set")
        
        return PostgresManager()
    
    elif storage_mode == 'local':
        if not os.environ.get('LOCAL_DATA_DIR'):
            logger.error("LOCAL_DATA_DIR environment variable is required for local storage")
            raise ValueError("LOCAL_DATA_DIR environment variable is not set")
        
        return LocalStorageManager()
    
    else:
        logger.error(f"Invalid STORAGE_MODE: {storage_mode}. Must be 'database' or 'local'")
        raise ValueError(f"Invalid STORAGE_MODE: {storage_mode}. Must be 'database' or 'local'")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Gemini Model Tuning with Flexible Text Storage')
    parser.add_argument('--setup-storage', action='store_true', help='Create the necessary database tables or local directories')
    parser.add_argument('--import-sample-data', action='store_true', help='Import sample text data into storage')
    parser.add_argument('--tune-model', action='store_true', help='Run the model tuning process')
    parser.add_argument('--test-model', action='store_true', help='Test the tuned model with sample prompts')
    
    args = parser.parse_args()
    
    # If no arguments provided, run all steps
    if not (args.setup_storage or args.import_sample_data or args.tune_model or args.test_model):
        args.setup_storage = True
        args.import_sample_data = True
        args.tune_model = True
        args.test_model = True
    
    return args

def check_environment_variables():
    """Check if required environment variables are set."""
    required_vars = ['GOOGLE_API_KEY', 'STORAGE_MODE']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set the following environment variables:")
        print("  GOOGLE_API_KEY: Your Google API key for Gemini")
        print("  STORAGE_MODE: Where to store training data ('database' or 'local')")
        
        storage_mode = os.environ.get('STORAGE_MODE', '').lower()
        if storage_mode == 'database' and not os.environ.get('DATABASE_URL'):
            print("  DATABASE_URL: PostgreSQL connection string (required for database storage)")
        elif storage_mode == 'local' and not os.environ.get('LOCAL_DATA_DIR'):
            print("  LOCAL_DATA_DIR: Directory for local text storage (required for local storage)")
        
        return False
    
    # Check mode-specific variables
    storage_mode = os.environ.get('STORAGE_MODE').lower()
    if storage_mode == 'database' and not os.environ.get('DATABASE_URL'):
        logger.error("DATABASE_URL environment variable is required for database storage")
        print("Error: DATABASE_URL environment variable is required for database storage")
        return False
    elif storage_mode == 'local' and not os.environ.get('LOCAL_DATA_DIR'):
        logger.error("LOCAL_DATA_DIR environment variable is required for local storage")
        print("Error: LOCAL_DATA_DIR environment variable is required for local storage")
        return False
    elif storage_mode not in ('database', 'local'):
        logger.error(f"Invalid STORAGE_MODE: {storage_mode}. Must be 'database' or 'local'")
        print(f"Error: Invalid STORAGE_MODE: {storage_mode}. Must be 'database' or 'local'")
        return False
    
    return True

def main():
    """Main function to run the Gemini text-only tuning process."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Check environment variables
    if not check_environment_variables():
        return 1
    
    try:
        # Create the appropriate storage manager
        logger.info(f"Creating storage manager for mode: {os.environ.get('STORAGE_MODE')}")
        storage_manager = create_storage_manager()
        
        # Setup storage if requested
        if args.setup_storage:
            logger.info("Setting up storage")
            if not storage_manager.setup_storage():
                logger.error("Failed to set up storage")
                return 1
        
        # Import sample data if requested
        if args.import_sample_data:
            logger.info("Importing sample text data")
            count = storage_manager.import_sample_data()
            if count == 0:
                logger.warning("No sample data was imported")
        
        # Initialize tuning manager
        if args.tune_model or args.test_model:
            logger.info("Initializing Gemini tuning manager")
            tuning_manager = GeminiTuningManager(storage_manager)
        
        # Tune the model if requested
        if args.tune_model:
            logger.info("Starting model tuning process")
            tuning_job = tuning_manager.tune_model()
            logger.info(f"Model tuning initiated successfully: {tuning_job.tuned_model.model}")
        
        # Test the model if requested
        if args.test_model:
            logger.info("Testing the tuned model")
            test_results = tuning_manager.test_model()
            logger.info(f"Model testing completed with {len(test_results)} prompts")
            print("\nTest Results:")
            for prompt, response in test_results.items():
                print(f"\nPrompt: {prompt}")
                print(f"Response: {response}")
        
        logger.info("Gemini text-only model tuning process completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
