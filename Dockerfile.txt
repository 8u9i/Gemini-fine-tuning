# Use an official Python 11 runtime as a parent image
FROM python:11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Make the start.sh script executable
RUN chmod +x start.sh

# Make port 80 available to the world outside this container (if your bot has a web interface or needs to listen)
# If your Discord bot only uses Discord's API, this line might not be strictly necessary,
# but it's good practice for web services.
EXPOSE 80

# Define environment variable (optional, for example if your bot needs a token)
# ENV DISCORD_BOT_TOKEN="your_token_here"

# Run the start.sh script when the container launches
CMD ["./start.sh"]
