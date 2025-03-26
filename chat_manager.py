import logging
from typing import Dict, List, Optional, Union, Any, Generator
from datetime import datetime

logger = logging.getLogger('GeminiTuner')

class ChatManager:
    """Manage chat interactions with Gemini models."""
    
    def __init__(self, model_manager):
        """
        Initialize chat manager.
        
        Args:
            model_manager: Model manager instance with initialized client
        """
        self.model_manager = model_manager
        self.active_chats = {}
        
    def create_chat(self, model_name: str = None, chat_id: str = None) -> str:
        """
        Create a new chat session.
        
        Args:
            model_name: Name of the model to use for chat
            chat_id: Optional custom ID for the chat session
            
        Returns:
            ID of the created chat session
        """
        if not self.model_manager.client:
            logger.error("Cannot create chat without initialized client.")
            return None
            
        # Use default model if not specified
        if not model_name:
            model_name = self.model_manager.config.get('model_name')
            
        # Generate a chat ID if not provided
        if not chat_id:
            chat_id = f"chat_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
        try:
            # Create a chat session
            chat = self.model_manager.client.chats.create(model=model_name)
            self.active_chats[chat_id] = {
                "chat": chat,
                "history": [],
                "model": model_name
            }
            logger.info(f"Created chat session {chat_id} with model {model_name}")
            return chat_id
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            return None
            
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
        if chat_id not in self.active_chats:
            logger.error(f"Chat session {chat_id} not found.")
            return None
            
        chat_session = self.active_chats[chat_id]
        chat = chat_session["chat"]
        
        try:
            # Add message to history
            chat_session["history"].append({"role": "user", "content": message})
            
            if stream:
                # Stream the response
                response_stream = chat.send_message_stream(message)
                
                # Define a generator to yield chunks
                def response_generator():
                    full_response = ""
                    for chunk in response_stream:
                        if chunk.text:
                            full_response += chunk.text
                            yield chunk.text
                    # Add to history after completion
                    chat_session["history"].append({"role": "model", "content": full_response})
                
                return response_generator()
            else:
                # Get complete response
                response = chat.send_message(message)
                # Add to history
                chat_session["history"].append({"role": "model", "content": response.text})
                return response.text
                
        except Exception as e:
            logger.error(f"Error sending message to chat {chat_id}: {e}")
            return None
            
    def get_history(self, chat_id: str) -> List[Dict[str, str]]:
        """
        Get the history of a chat session.
        
        Args:
            chat_id: ID of the chat session
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        if chat_id not in self.active_chats:
            logger.error(f"Chat session {chat_id} not found.")
            return []
            
        return self.active_chats[chat_id]["history"]
        
    def list_chats(self) -> List[Dict[str, Any]]:
        """
        List all active chat sessions.
        
        Returns:
            List of dictionaries with chat information
        """
        chat_info = []
        for chat_id, session in self.active_chats.items():
            info = {
                "id": chat_id,
                "model": session["model"],
                "messages": len(session["history"]),
                "created": chat_id.split("_")[1] if "_" in chat_id else "unknown"
            }
            chat_info.append(info)
            
        return chat_info
        
    def delete_chat(self, chat_id: str) -> bool:
        """
        Delete a chat session.
        
        Args:
            chat_id: ID of the chat session
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if chat_id not in self.active_chats:
            logger.error(f"Chat session {chat_id} not found.")
            return False
            
        try:
            del self.active_chats[chat_id]
            logger.info(f"Deleted chat session {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting chat session {chat_id}: {e}")
            return False
