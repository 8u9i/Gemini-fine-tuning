Enhanced Gemini Tuner

A powerful and user-friendly tool for fine-tuning Google's Gemini models with your own data. This application provides multiple interfaces (GUI, interactive CLI, and command-line) to make model tuning accessible to users of all technical levels.
Gemini Tuner
Features

Multiple User Interfaces:
Graphical User Interface (GUI) for intuitive operation
Interactive command-line mode with guided workflows
Traditional command-line arguments for automation and scripting
Model Training:
Fine-tune Gemini models with your custom data
Support for various training parameters (epochs, batch size, learning rate)
Progress visualization during training
Data Processing:
Convert text files into training examples
Batch processing of multiple input files
Customizable format settings for different data structures
Chat Functionality:
Create and manage chat sessions with Gemini models
Support for streaming responses
Chat history management
Model Management:
List, test, and delete tuned models
Preview model responses before deployment
Configuration profiles for different projects
Installation

Prerequisites
Python 3.8 or higher
Google AI API key (get one from Google AI Studio )
Install from Source
bash
# Clone the repository or download the files
git clone https://github.com/yourusername/gemini-tuner.git
cd gemini-tuner

# Install required packages
pip install -r requirements.txt
Manual Installation
Download the enhanced_gemini_tuner.py and chat_manager.py files
Install required packages:
bash
pip install google-generativeai python-dotenv tqdm PySimpleGUI
Quick Start

GUI Mode
bash
python enhanced_gemini_tuner.py --gui
Interactive CLI Mode
bash
python enhanced_gemini_tuner.py --interactive
Command-line Mode
bash
# Process files and create training data
python enhanced_gemini_tuner.py process --input-files data/*.txt --output training_data.json

# Train a model with the processed data
python enhanced_gemini_tuner.py train --data training_data.json --model gemini-2.0-flash

# Test a tuned model
python enhanced_gemini_tuner.py test --model-id your-model-id --input "Your test prompt here"
Configuration

API Key Setup
Create a .env file in the same directory as the script:
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash
GEMINI_EPOCHS=5
GEMINI_BATCH_SIZE=4
GEMINI_LEARNING_RATE=1.0
Alternatively, use the setup command:
bash
python enhanced_gemini_tuner.py setup --create-config
Configuration Profiles
You can create multiple configuration profiles for different projects:
bash
python enhanced_gemini_tuner.py setup --create-profile my_project
Usage Examples

Processing Training Data
python
from enhanced_gemini_tuner import Config, DataProcessor

config = Config() 
processor = DataProcessor(config)

# Process a single file
examples = processor.process_file("data.txt")

# Process multiple files
examples = processor.process_files(["file1.txt", "file2.txt"])

# Save processed data
processor.save_examples(examples, "training_data.json")
Training a Model
python
from enhanced_gemini_tuner import Config, ModelManager

config = Config()
model_manager = ModelManager(config)

# Load training data
with open("training_data.json", "r") as f:
    training_data = json.load(f)

# Train model
model_id = model_manager.train_model(
    training_data=training_data,
    model_name="gemini-2.0-flash",
    epochs=5,
    batch_size=4,
    learning_rate_multiplier=1.0
)

print(f"Trained model ID: {model_id}")
Using Chat Functionality
python
from enhanced_gemini_tuner import Config, ModelManager

config = Config()
model_manager = ModelManager(config)

# Create a chat session
chat_id = model_manager.create_chat()

# Send a message and get a response
response = model_manager.send_message(chat_id, "Tell me about the Gemini API")
print(response)

# Get streaming responses
for chunk in model_manager.send_message(chat_id, "Write a poem about AI", stream=True):
    print(chunk, end="")

# View chat history
history = model_manager.get_chat_history(chat_id)
Mobile Usage

You can run the Enhanced Gemini Tuner on mobile devices using:
Android (Termux)
Install Termux from Google Play Store or F-Droid
Set up Python environment: pkg install python
Install required packages: pip install google-generativeai python-dotenv tqdm
Run in interactive mode: python enhanced_gemini_tuner.py --interactive
iOS (iSH or a-Shell)
Install iSH or a-Shell from App Store
Set up Python and install required packages
Run in interactive CLI mode
Troubleshooting

Common Issues
API Key Issues: Ensure your API key is correctly set in the .env file or passed as an environment variable
Import Errors: Make sure all required packages are installed
Permission Errors: Ensure the script has write permissions in the current directory
Debug Mode
Run with debug logging enabled:
bash
python enhanced_gemini_tuner.py --debug
License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Google Generative AI team for the Gemini API
Contributors to the Python libraries used in this project
