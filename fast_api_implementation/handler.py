from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.messages import BaseMessage
from langchain.schema import LLMResult
from typing import Dict, List, Any

from langchain_core.outputs import ChatGenerationChunk, GenerationChunk

# Create custom callback handler class
class MyCustomHandler(BaseCallbackHandler):
    def __init__(self, queue) -> None:
        super().__init__()
        # Initiate provider to streamer queue as input
        self._queue = queue
        # Define stop signal added to queue in case of the last token
        self._stop_signal = None
        print('Custom handler initialized')
        
    # Add new token to queue on its arrival
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self._queue.put(token)
        
    
    # On start, print a starting message 
    def on_llm_start( 
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        print("Generation  has started")
        
    # On receiving last token, we add the stop signal
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM stops running"""
        print('\nGeneration concluded')
        self._queue.put(self._stop_signal)